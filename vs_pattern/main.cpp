#include <iostream>
#include <opencv2/opencv.hpp>
#include "data.cpp"
#include <ctime>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
using namespace cv;
namespace py = pybind11;

// 是否打印日志
bool Log_Print;
// 图像存储根目录
const string Base_Dir = "img/", Template_Dir = Base_Dir + "template/", Stage_Dir = Base_Dir + "stage/";
const string Split_Dir = Base_Dir + "split/", Mark_Dir = Base_Dir + "mark/";
// 17个模板图案
const int Template_Type = 7, Template_Size = 13, Thick = 3;
vector<Mat> Template[Template_Type];
// 上下左右4个方向
const int Direct[][2] = { {-1, 0}, {0, 1}, {1, 0}, {0, -1} };
// 阶段
char Stage_Num;
// 灰度图和hsv图
Mat gray_img, hsv_img;
// 左右粗略边界，背景亮度阈值
const int Left_Border = 400, Right_Border = 1500, Bright_Above_Thresh = 40;
// 开闭运算窗口大小
const int Open_Win = 2, Close_Win = 2, Close_Win_Merge = 12;
// 最大连通域的搜寻窗口，裁剪时保留的边缘大小
const int Domain_Radius = 20, Edge = 7;
// 宽高上限
const int Max_H = 1080, Max_W = 1920;
int Min_X, Max_X, Min_Y, Max_Y;
// 每个像素的连通域编号，连通域边界
int Region_Num;
int Pixel_Region[Max_H][Max_W];
Rect Region_Border[Max_H * Max_W];
// 每个符号的边界，识别结果
const int Pattern_Size = 64;
const char* Character = "SOCLT+-";
Rect Pattern_Border[Pattern_Size][Pattern_Size];
int Region_Result[Pattern_Size][Pattern_Size];
char Character_Result[Pattern_Size][Pattern_Size];
// 符号的边长，间距
const int Pattern_Radius = 12, Pattern_Margin = 2;
// 裁剪之后的偏移(X都是垂直距离，Y是水平距离)
int Offset_X = 0, Offset_Y = 0;
// 相机参数
const double P[3][3] = {
    {1.55668627e+03, 0.00000000e+00, 9.69756395e+02},
    {0.00000000e+00, 1.55968033e+03, 5.93634464e+02},
    {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
const double A = 18.5, B = 0.05, L = 60, F = 35.0, EPS = 5e-1, Min_Z = 0.0, Max_Z = 15.0;
// 符号坐标，四周符号坐标平均值
pair<double, double> Coord[Pattern_Size][Pattern_Size], Avg_Coord[Pattern_Size][Pattern_Size];
vector<vector<vector<double>>> World_Coord(Pattern_Size, vector<vector<double>>(Pattern_Size, vector<double>(3, 0.0)));

void init(bool& flag) {
    Log_Print = flag;
    Stage_Num = '0';
    Region_Num = 0;
    Offset_X = Offset_Y = 0;
    memset(Pixel_Region, 0, sizeof(Pixel_Region));
    memset(Region_Border, 0, sizeof(Region_Border));
    memset(Pattern_Border, 0, sizeof(Pattern_Border));
    memset(Region_Result, 0, sizeof(Region_Result));
    memset(Character_Result, 0, sizeof(Character_Result));
}

void update_contour(int flag, int x=0, int y=0) {
    if (flag == 0)
        Min_X = Max_H, Max_X = 0, Min_Y = Max_W, Max_Y = 0;
    else {
        Min_X = min(Min_X, x);
        Max_X = max(Max_X, x);
        Min_Y = min(Min_Y, y);
        Max_Y = max(Max_Y, y);
    }
}

// 制作13x13的模板图案
void make_template() {
    Size size(Template_Size, Template_Size);
    Mat pattern = Mat::zeros(size, CV_8UC1);
    // S
    for (int i = 0; i < pattern.rows; ++i)
        for (int j = 0; j < pattern.cols; ++j)
            pattern.at<uchar>(i, j) = 255;
    Template[0].push_back(pattern.clone());
    // O
    for (int i = Thick; i < Template_Size - Thick; ++i)
        for (int j = Thick; j < Template_Size - Thick; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[1].push_back(pattern.clone());
    // C:4
    for (int i = Thick; i < Template_Size - Thick; ++i)
        for (int j = Template_Size - Thick; j < Template_Size; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[2].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
        Template[2].push_back(pattern.clone());
    }
    // L:4
    for (int i = 0; i < Template_Size - Thick; ++i)
        for (int j = Template_Size - Thick; j < Template_Size; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[3].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_COUNTERCLOCKWISE);
        Template[3].push_back(pattern.clone());
    }
    // T:4
    for (int i = Thick; i < Template_Size; ++i) {
        for (int j = 0; j < Thick; ++j)
            pattern.at<uchar>(i, j) = 0;
        for (int j = Template_Size / 2 - 1; j < Template_Size / 2 + 2; ++j)
            pattern.at<uchar>(i, j) = 255;
    }
    Template[4].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
        Template[4].push_back(pattern.clone());
    }
    // +
    for (int i = 0; i < Template_Size; ++i) {
        if (i >= Template_Size / 2 - 1 && i < Template_Size / 2 + 2)
            continue;
        for (int j = 0; j < Thick; ++j)
            pattern.at<uchar>(i, j) = 0;
        for (int j = Template_Size / 2 - 1; j < Template_Size / 2 + 2; ++j)
            pattern.at<uchar>(i, j) = 255;
    }
    Template[5].push_back(pattern.clone());
    // -:2
    for (int i = 0; i < Template_Size; ++i) {
        if (i >= Template_Size / 2 - 1 && i < Template_Size / 2 + 2)
            continue;
        for (int j = Template_Size / 2 - 1; j < Template_Size / 2 + 2; ++j)
            pattern.at<uchar>(i, j) = 0;
    }
    Template[6].push_back(pattern.clone());
    rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
    Template[6].push_back(pattern.clone());
    // 保存
    for (int i = 0; i < Template_Type; ++i)
        for (int j = 0; j < Template[i].size(); ++j)
            imwrite(Template_Dir + to_string(i) + "_" + to_string(j) + ".jpg", Template[i][j]);
}

// 读取灰度图和hsv图
void read_gray_hsv(const string& img_name) {
    Mat color_img = imread(Base_Dir + img_name);
    cvtColor(color_img, gray_img, COLOR_BGR2GRAY);
    imwrite(Stage_Dir + (++Stage_Num) + "_gray_img.jpg", gray_img);
    cvtColor(color_img, hsv_img, COLOR_BGR2HSV);
    imwrite(Stage_Dir + (++Stage_Num) + "_hsv_img.jpg", hsv_img);
    if (Log_Print)
        printf("read_gray_hsv done\n");
}

// 裁剪两边大片无关区域，去除亮度低于40的背景像素，并二值化
void bright_above() {
    Rect area(Left_Border, 0, Right_Border - Left_Border + 1, gray_img.rows);
    gray_img = gray_img(area);
    hsv_img = hsv_img(area);
    Offset_X += Left_Border;
    for (int i = 0; i < gray_img.rows; ++i)
        for (int j = 0; j < gray_img.cols; ++j)
            if (hsv_img.at<Vec3b>(i, j)[2] < Bright_Above_Thresh * 255 / 100.0)
                gray_img.at<uchar>(i, j) = 0;
            else
                gray_img.at<uchar>(i, j) = 255;
    imwrite(Stage_Dir + (++Stage_Num) + "_bright_above.jpg", gray_img);
    if (Log_Print)
        printf("bright_above done\n");
}

// 保留最大的连通域，为简化起见默认图像中间的部分属于该连通域
vector<vector<bool>> find_biggest() {
    int row = gray_img.rows, col = gray_img.cols, x, y = col / 2;
    // 寻找到某一点
    for (int i = row / 2 - Domain_Radius; i < row / 2 + Domain_Radius; ++i)
        if (gray_img.at<uchar>(i, col / 2) == 255) {
            x = i;
            break;
        }
    // 遍历连通域
    queue<pair<int, int>> candidate;
    vector<vector<bool>> flag(row, vector<bool>(col, false));
    candidate.push(make_pair(x, y));
    flag[x][y] = true;
    while (!candidate.empty()) {
        pair<int, int> cur = candidate.front();
        candidate.pop();
        x = cur.first, y = cur.second;
        for (auto d : Direct) {
            int nx = x + d[0], ny = y + d[1];
            if (gray_img.at<uchar>(nx, ny) == 255 && flag[nx][ny] == false) {
                candidate.push(make_pair(nx, ny));
                flag[nx][ny] = true;
            }
        }
    }
    return flag;
}

// 提取粗略边界(稍微留些边缘)
void remain_biggest(vector<vector<bool>>& flag) {
    update_contour(0);
    int row = gray_img.rows, col = gray_img.cols;
    // 填充四周的背景
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (flag[i][j]) {
                update_contour(1, i, j);
                break;
            }
            hsv_img.at<Vec3b>(i, j)[2] = 0;
        }
        for (int j = col - 1; j >= 0; --j) {
            if (flag[i][j]) {
                update_contour(1, i, j);
                break;
            }
            hsv_img.at<Vec3b>(i, j)[2] = 0;
        }
    }
    for (int j = 0; j < col; ++j) {
        for (int i = 0; i < row; ++i) {
            if (flag[i][j]) {
                update_contour(1, i, j);
                break;
            }
            hsv_img.at<Vec3b>(i, j)[2] = 0;
        }
        for (int i = row - 1; i >= 0; --i) {
            if (flag[i][j]) {
                update_contour(1, i, j);
                break;
            }
            hsv_img.at<Vec3b>(i, j)[2] = 0;
        }
    }
    // 剪裁，保留些边缘
    Rect area(Min_Y - Edge, Min_X - Edge, Max_Y - Min_Y + 1 + 2 * Edge, Max_X - Min_X + 1 + 2 * Edge);
    gray_img = hsv_img(area);
    Offset_X += Min_Y - Edge, Offset_Y += Min_X - Edge;
    cvtColor(gray_img, gray_img, COLOR_HSV2BGR);
    cvtColor(gray_img, gray_img, COLOR_BGR2GRAY);
}

// 先做闭运算，保留最大的连通域，提取粗略边界(稍微留些边缘)
void extract_border() {
    Mat kernel = getStructuringElement(MORPH_RECT, Size(Close_Win_Merge, Close_Win_Merge));
    morphologyEx(gray_img, gray_img, MORPH_CLOSE, kernel);
    // 保留最大的连通域，为简化起见默认图像中间的部分属于该连通域
    vector<vector<bool>> flag= find_biggest();
    // 提取粗略边界(稍微留些边缘)
    remain_biggest(flag);

    imwrite(Stage_Dir + (++Stage_Num) + "_extract_border.jpg", gray_img);
    if (Log_Print)
        printf("extract_border done\n");
}

// 局部均值二值化，再进行开闭运算去噪
void avg_thresh() {
    // 局部均值阈值化
    Mat tmp = gray_img.clone();
    for (int x = Edge; x < gray_img.rows - Edge; ++x)
        for (int y = Edge; y < gray_img.cols - Edge; ++y)
            if (gray_img.at<uchar>(x, y) > 0) {
                int sum = 0, cnt = 0;
                for (int i = -Edge; i <= Edge; ++i)
                    for (int j = -Edge; j <= Edge; ++j)
                        if (tmp.at<uchar>(x + i, y + j) > 0) {
                            sum += tmp.at<uchar>(x + i, y + j);
                            ++cnt;
                        }
                gray_img.at<uchar>(x, y) = gray_img.at<uchar>(x, y) < (double)sum / cnt ? 0 : 255;
            }
    // 开闭运算去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(Open_Win, Open_Win));
    morphologyEx(gray_img, gray_img, MORPH_OPEN, kernel);
    kernel = getStructuringElement(MORPH_RECT, Size(Close_Win, Close_Win));
    morphologyEx(gray_img, gray_img, MORPH_CLOSE, kernel);

    imwrite(Stage_Dir + (++Stage_Num) + "_avg_thresh.jpg", gray_img);
    if (Log_Print)
        printf("avg_thresh done\n");
}

// 分解成一个个连通域
void split_domain(const int min_size) {
    vector<vector<bool>> flag(gray_img.rows, vector<bool>(gray_img.cols, false));
    for (int i = 1; i < gray_img.rows - 1; ++i)
        for (int j = 1; j < gray_img.cols - 1; ++j)
            if (gray_img.at<uchar>(i, j) == 255 && flag[i][j] == false) {
                update_contour(0);
                ++Region_Num;
                int cnt = 0;
                queue<pair<int, int>> candidate, all_candidate;
                candidate.push(make_pair(i, j));
                all_candidate.push(make_pair(i, j));
                flag[i][j] = true;
                // 给连通域的每个元素相同的编号
                while (!candidate.empty()) {
                    pair<int, int> cur = candidate.front();
                    candidate.pop();
                    int x = cur.first, y = cur.second;
                    update_contour(1, x, y);
                    Pixel_Region[x][y] = Region_Num;
                    ++cnt;
                    for (auto d : Direct) {
                        int nx = x + d[0], ny = y + d[1];
                        if (gray_img.at<uchar>(nx, ny) == 255 && flag[nx][ny] == false) {
                            candidate.push(make_pair(nx, ny));
                            all_candidate.push(make_pair(nx, ny));
                            flag[nx][ny] = true;
                        }
                    }
                }
                if (cnt > min_size) {
                    Region_Border[Region_Num] = Rect(Min_Y, Min_X, Max_Y - Min_Y + 1, Max_X - Min_X + 1);
                    continue;
                }
                // 连通域过小则去除
                --Region_Num;
                while (!all_candidate.empty()) {
                    pair<int, int> cur = all_candidate.front();
                    all_candidate.pop();
                    gray_img.at<uchar>(cur.first, cur.second) = 0;
                    Pixel_Region[cur.first][cur.second] = 0;
                }
            }

    imwrite(Stage_Dir + (++Stage_Num) + "_split_domain.jpg", gray_img);
    if (Log_Print)
        printf("split_domain done\n");
}

// 识别符号
int recognize_pattern(Mat& pattern) {
    int row = pattern.rows, col = pattern.cols;
    if (row != Template_Size) {
        // 要处理的上下行数
        int up = abs(row - Template_Size) / 2, down = up;
        if ((row - Template_Size) % 2 == 1) {
            int up_num = 0, down_num = 0;
            for (int j = 0; j < col; ++j) {
                if (pattern.at<uchar>(0, j) == 255)
                    ++up_num;
                if (pattern.at<uchar>(row - 1, j) == 255)
                    ++down_num;
            }
            (row > Template_Size) ^ (up_num > down_num) ? ++up : ++down;
        }
        if (row > Template_Size)
            pattern = pattern(Rect(0, up, col, Template_Size));
        else {
            Mat tmp = pattern.clone();
            pattern = Mat(Template_Size, col, CV_8UC1);
            for (int j = 0; j < col; ++j) {
                for (int i = 0; i < up; ++i)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(0, j);
                for (int i = up; i < up + row; ++i)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(i - up, j);
                for (int i = up + row; i < Template_Size; ++i)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(row - 1, j);
            }
        }
    }
    if (col != Template_Size) {
        // 要处理的左右列数
        int left = abs(col - Template_Size) / 2, right = left;
        if ((col - Template_Size) % 2 == 1) {
            int left_num = 0, right_num = 0;
            for (int i = 0; i < Template_Size; ++i) {
                if (pattern.at<uchar>(i, 0) == 255)
                    ++left_num;
                if (pattern.at<uchar>(i, col - 1) == 255)
                    ++right_num;
            }
            (col > Template_Size) ^ (left_num > right_num) ? ++left : ++right;
        }
        if (col > Template_Size)
            pattern = pattern(Rect(left, 0, Template_Size, Template_Size));
        else {
            Mat tmp = pattern.clone();
            pattern = Mat(Template_Size, Template_Size, CV_8UC1);
            for (int i = 0; i < Template_Size; ++i) {
                for (int j = 0; j < left; ++j)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(i, 0);
                for (int j = left; j < left + col; ++j)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(i, j - left);
                for (int j = left + col; j < Template_Size; ++j)
                    pattern.at<uchar>(i, j) = tmp.at<uchar>(i, col - 1);
            }
        }
    }
    // 再跟模板一个个比较
    int max_area = 0, max_index = -1;
    for (int h = 0; h < Template_Type; ++h)
        for (int k = 0; k < Template[h].size(); ++k) {
            int same = 0;
            for (int i = 0; i < Template_Size; ++i)
                for (int j = 0; j < Template_Size; ++j)
                    if (pattern.at<uchar>(i, j) == Template[h][k].at<uchar>(i, j))
                        ++same;
            if (same > max_area) {
                max_area = same;
                max_index = h;
            }
        }
    // printf("%d\t%d\n", max_index, max_area);
    return max_index;
}

// 寻找第(0, 0)个符号，按副对角线方向扫描
bool search_first() {
    for (int k = 0; k < min(gray_img.rows, gray_img.cols); ++k)
        for (int i = 0; i <= k; ++i)
            if (gray_img.at<uchar>(i, k - i) == 255) {
                Pattern_Border[0][0] = Region_Border[Pixel_Region[i][k - i]];
                Mat pattern = gray_img(Pattern_Border[0][0]);
                imwrite(Split_Dir + "0_0.jpg", pattern);
                Region_Result[0][0] = recognize_pattern(pattern);
                if (Region_Result[0][0] == -1) {
                    printf("Recognize domain (0, 0) failed！！！\n");
                    return false;
                } else {
                    Character_Result[0][0] = Character[Region_Result[0][0]];
                    return true;
                }
            }
    return true;
}

// 依次寻找每个符号
bool search_next(const int row, const int col) {
    // 1、根据左边的(即前一个，若不存在，则根据上边的)符号确定该符号的大致范围
    int left, up, right, down;
    if (col == 0) {
        Rect pre_region = Pattern_Border[row - 1][col];
        up = pre_region.y + pre_region.height, down = up + Pattern_Radius + Pattern_Margin * 2 - 1;
        left = pre_region.x - Pattern_Margin, right = pre_region.x + pre_region.width + Pattern_Margin - 1;
    } else {
        Rect pre_region = Pattern_Border[row][col - 1];
        up = pre_region.y - Pattern_Margin, down = pre_region.y + pre_region.height + Pattern_Margin - 1;
        left = pre_region.x + pre_region.width, right = left + Pattern_Radius + Pattern_Margin * 2 - 1;
    }
    
    // 2、根据大致范围确定具体的连通域
    map<int, int> all_times;
    for (int i = up; i <= down; ++i)
        for (int j = left; j <= right; ++j) {
            int num = Pixel_Region[i][j];
            if (num != 0) {
                if (all_times.count(num) == 0)
                    all_times[num] = 1;
                else
                    ++all_times[num];
            }
        }
    int region_num = 0, region_times = 0;
    for (auto cur_times : all_times)
        if (cur_times.second > region_times) {
            region_num = cur_times.first;
            region_times = cur_times.second;
        }
    if (region_num == 0) {
        printf("Search domain (%d, %d) failed！！！\n", row, col);
        printf("(%d %d %d %d)\n", up, down, left, right);
        for (int ww = up; ww <= down; ++ww) {
            for (int ee = left; ee <= right; ++ee)
                printf("%d ", Pixel_Region[ww][ee]);
            printf("\n");
        }
        return false;
    }
    Rect region = Region_Border[region_num];

    // 3、连通域太大，需要切割。竖着一刀，横着一刀
    bool has_cut = false;
    int thresh = row > 0 ? Pattern_Border[row - 1][col].width / 2 : Pattern_Radius / 2;
    // 竖着切
    if (region.x + region.width > right + thresh) {
        has_cut = true;
        // 切断
        for (int i = up; i <= down; ++i)
            if (Pixel_Region[i][right - 1] == region_num) {
                Pixel_Region[i][right - 1] = 0;
                gray_img.at<uchar>(i, right - 1) = 0;
            }
        // 建立新的连通域
        for (int i = 0; i <= down; ++i)
            if (Pixel_Region[i][right] == region_num) {
                update_contour(0);
                ++Region_Num;
                Pixel_Region[i][right] = Region_Num;
                queue<pair<int, int>> candidate;
                candidate.push(make_pair(i, right));
                while (!candidate.empty()) {
                    pair<int, int> cur = candidate.front();
                    candidate.pop();
                    int x = cur.first, y = cur.second;
                    update_contour(1, x, y);
                    for (auto d : Direct) {
                        int nx = x + d[0], ny = y + d[1];
                        if (Pixel_Region[nx][ny] == region_num) {
                            candidate.push(make_pair(nx, ny));
                            Pixel_Region[nx][ny] = Region_Num;
                        }
                    }
                }
                Region_Border[Region_Num] = Rect(Min_Y, Min_X, Max_Y - Min_Y + 1, Max_X - Min_X + 1);
                break;
            }
    }
    thresh = col > 0 ? Pattern_Border[row][col - 1].height / 2 : Pattern_Radius / 2;
    // 横着切
    if (region.y + region.height > down + thresh) {
        has_cut = true;
        // 切断
        for (int j = left; j <= right; ++j)
            if (Pixel_Region[down - 1][j] == region_num) {
                Pixel_Region[down - 1][j] = 0;
                gray_img.at<uchar>(down - 1, j) = 0;
            }
        // 建立新的连通域
        for (int j = left; j <= right; ++j)
            if (Pixel_Region[down][j] == region_num) {
                update_contour(0);
                ++Region_Num;
                Pixel_Region[down][j] = Region_Num;
                queue<pair<int, int>> candidate;
                candidate.push(make_pair(down, j));
                while (!candidate.empty()) {
                    pair<int, int> cur = candidate.front();
                    candidate.pop();
                    int x = cur.first, y = cur.second;
                    update_contour(1, x, y);
                    for (auto d : Direct) {
                        int nx = x + d[0], ny = y + d[1];
                        if (Pixel_Region[nx][ny] == region_num) {
                            candidate.push(make_pair(nx, ny));
                            Pixel_Region[nx][ny] = Region_Num;
                        }
                    }
                }
                Region_Border[Region_Num] = Rect(Min_Y, Min_X, Max_Y - Min_Y + 1, Max_X - Min_X + 1);
                break;
            }
    }

    // 4、重新计算当前连通域
    if (has_cut) {
        update_contour(0);
        for (int i = up; i <= down + Pattern_Radius / 2; ++i)
            for (int j = left; j <= right + Pattern_Radius / 2; ++j)
                if (Pixel_Region[i][j] == region_num)
                    update_contour(i, j);
        region = Rect(Min_Y, Min_X, Max_Y - Min_Y + 1, Max_X - Min_X + 1);
        Region_Border[region_num] = region;
    }
    
    // 5、连通域过大或过小则参考邻居大小进行切割
    if (region.height < Pattern_Radius * 2 / 3 || region.height > Pattern_Radius * 4 / 3
        || region.width < Pattern_Radius * 2 / 3 || region.width > Pattern_Radius * 4 / 3) {
        if (row > 0) {
            if (row == 1)
                region.x = Pattern_Border[row - 1][col].x;
            if (row > 1)
                region.x = 2 * Pattern_Border[row - 1][col].x - Pattern_Border[row - 2][col].x;
            region.width = Pattern_Border[row - 1][col].width;
        }
        if (col > 0) {
            if (col == 1)
                region.y = Pattern_Border[row][col - 1].y;
            if (col > 1)
                region.y = 2 * Pattern_Border[row][col - 1].y - Pattern_Border[row][col - 2].y;
            region.height = Pattern_Border[row][col - 1].height;
        }
        if (row == 0) {
            region.x = Pattern_Border[row][col - 1].x + Pattern_Border[row][col - 1].width + Pattern_Margin - 1;
            region.width = region.height;
        }
        if (col == 0) {
            region.y = Pattern_Border[row - 1][col].y + Pattern_Border[row - 1][col].height + Pattern_Margin - 1;
            region.height = region.width;
        }
    }
    Pattern_Border[row][col] = region;
    //ofstream out("rect_cpp.txt", ios::app);
    //out << region.y << "\t" << region.y + region.height - 1 << "\t" << region.x << "\t" << region.x + region.width - 1 << endl;

    // 6、提取符号进行识别
    Mat pattern = gray_img(region);
    imwrite(Split_Dir + to_string(row) + "_" + to_string(col) + ".jpg", pattern);
    Region_Result[row][col] = recognize_pattern(pattern);
    if (Region_Result[row][col] == -1) {
        printf("Recognize domain (%d, %d)failed！！！\n", row, col);
        return false;
    }
    else {
        Character_Result[row][col] = Character[Region_Result[row][col]];
        return true;
    }
}

// 寻找每一个符号
void search_pattern() {
    // 先寻找第(0, 0)个符号
    if (search_first())
        for (int i = 0; i < Pattern_Size; ++i)
            for (int j = 0; j < Pattern_Size; ++j) {
                // 依次寻找每个符号
                if ((i || j) && !search_next(i ,j))
                    return;
            }
    if (Log_Print)
        printf("search_pattern done\n");
}

// 标记识别错误的地方并打印结果
void mark_print(const string& img_name) {
    for (int i = 0; i < gray_img.rows; ++i)
        for (int j = 0; j < gray_img.cols; ++j)
            if (gray_img.at<uchar>(i, j) == 255)
                gray_img.at<uchar>(i, j) = 127;
    int right = 0;
    for (int i = 0; i < Pattern_Size; ++i)
        for (int j = 0; j < Pattern_Size; ++j)
            if (Region_Result[i][j] == Right_Answer[i][j])
                ++right;
            else {
                Rect region = Pattern_Border[i][j];
                for (int x = 0; x < region.height; ++x)
                    for (int y = 0; y < region.width; ++y)
                        if (gray_img.at<uchar>(x + region.y, y + region.x) == 127)
                            gray_img.at<uchar>(x + region.y, y + region.x) = 255;
            }
    imwrite(Mark_Dir + img_name, gray_img);

    if (Log_Print) {
        for (int i = 0; i < Pattern_Size; ++i) {
            for (int j = 0; j < Pattern_Size; ++j)
                printf("%c ", Character_Result[i][j]);
            printf("\n");
        }
    }
    printf("The accuracy of %s is %.3f%%\n", img_name.c_str(), (double)right / Pattern_Size / Pattern_Size * 100);
}

// 计算每个符号的坐标以及四周符号坐标的平均值
void cacul_coord(const string& img_name) {
    for (int i = 0; i < Pattern_Size; ++i)
        for (int j = 0; j < Pattern_Size; ++j) {
            Rect region = Pattern_Border[i][j];
            Coord[i][j] = make_pair(region.x - 0.5 + region.width / 2.0 + Offset_X, region.y - 0.5 + region.height / 2.0 + Offset_Y);
        }
    for (int i = 1; i < Pattern_Size - 1; ++i)
        for (int j = 1; j < Pattern_Size - 1; ++j)
            Avg_Coord[i][j] = make_pair((Coord[i - 1][j].first + Coord[i + 1][j].first + Coord[i][j - 1].first + Coord[i][j + 1].first) / 4.0,
                (Coord[i - 1][j].second + Coord[i + 1][j].second + Coord[i][j - 1].second + Coord[i][j + 1].second) / 4.0);
    if ("standard.jpg" == img_name) {
        // 标准图案
        for (int i = 0; i < Pattern_Size; ++i)
            for (int j = 0; j < Pattern_Size; ++j)
                if ((i + j) % 2 == 1) {
                    World_Coord[i][j][0] = (L - B) * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = -(L - B) * (Coord[i][j].second - P[1][2]) / P[1][1];
                    World_Coord[i][j][2] = B;
                }
                else {
                    World_Coord[i][j][0] = L * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = -L * (Coord[i][j].second - P[1][2]) / P[1][1];
                }
    } else {
        // 计算世界坐标
        for (int i = 1; i < Pattern_Size - 1; ++i)
            for (int j = 1; j < Pattern_Size - 1; ++j) {
                if ((i + j) % 2 == 1) {
                    // 取平均值
                    if (fabs(Coord[i][j].first - Avg_Coord[i][j].first) > EPS || fabs(Coord[i][j].second - Avg_Coord[i][j].second) > EPS) {
                        double H1 = Coord[i][j].first - Avg_Coord[i][j].first;
                        double H2 = L * (Avg_Coord[i][j].first - P[0][2]) - P[0][0] * Standard_Coord[i][j][0];
                        double H3 = L * (P[0][2] - Coord[i][j].first) + B * (Avg_Coord[i][j].first - P[0][2]) + P[0][0] * Standard_Coord[i][j][0];
                        double H4 = B * (L * (Avg_Coord[i][j].first - P[0][2]) - P[0][0] * Standard_Coord[i][j][0]);
                        double K1 = Coord[i][j].second - Avg_Coord[i][j].second;
                        double K2 = L * (Avg_Coord[i][j].second - P[1][2]) + P[1][1] * Standard_Coord[i][j][1];
                        double K3 = L * (P[1][2] - Coord[i][j].second) + B * (Avg_Coord[i][j].second - P[1][2]) - P[1][1] * Standard_Coord[i][j][1];
                        double K4 = B * (L * (Avg_Coord[i][j].second - P[1][2]) + P[1][1] * Standard_Coord[i][j][1]);
                        double W1 = H1 * K3 - H3 * K1;
                        double W2 = H4 * K1 - H3 * K2 + H2 * K3 - H1 * K4;
                        double W3 = H4 * K2 - H2 * K4;
                        double delta = W2 * W2 - 4 * W1 * W3;
                        if (delta >= 0) {
                            World_Coord[i][j][2] = (sqrt(delta) - W2) / (2 * W1);
                            if (World_Coord[i][j][2] < Min_Z || World_Coord[i][j][2] > Max_Z)
                                World_Coord[i][j][2] = Standard_Coord[i][j][2];
                        }
                    }
                    World_Coord[i][j][0] = (L - World_Coord[i][j][2]) * (Avg_Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = (L - World_Coord[i][j][2]) * (Avg_Coord[i][j].second - P[1][2]) / P[1][1];
                } else {
                    // 取当前点
                    if (fabs(Coord[i][j].first - Avg_Coord[i][j].first) > EPS || fabs(Coord[i][j].second - Avg_Coord[i][j].second) > EPS) {
                        double H1 = Avg_Coord[i][j].first - Coord[i][j].first;
                        double H2 = L * (Coord[i][j].first - P[0][2]) - P[0][0] * Standard_Coord[i][j][0];
                        double H3 = L * (P[0][2] - Avg_Coord[i][j].first) + B * (Coord[i][j].first - P[0][2]) + P[0][0] * Standard_Coord[i][j][0];
                        double H4 = B * (L * (Coord[i][j].first - P[0][2]) - P[0][0] * Standard_Coord[i][j][0]);
                        double K1 = Avg_Coord[i][j].second - Coord[i][j].second;
                        double K2 = L * (Coord[i][j].second - P[1][2]) + P[1][1] * Standard_Coord[i][j][1];
                        double K3 = L * (P[1][2] - Avg_Coord[i][j].second) + B * (Coord[i][j].second - P[1][2]) - P[1][1] * Standard_Coord[i][j][1];
                        double K4 = B * (L * (Coord[i][j].second - P[1][2]) + P[1][1] * Standard_Coord[i][j][1]);
                        double W1 = H1 * K3 - H3 * K1;
                        double W2 = H4 * K1 - H3 * K2 + H2 * K3 - H1 * K4;
                        double W3 = H4 * K2 - H2 * K4;
                        double delta = W2 * W2 - 4 * W1 * W3;
                        if (delta >= 0) {
                            World_Coord[i][j][2] = (sqrt(delta) - W2) / (2 * W1);
                            if (World_Coord[i][j][2] < Min_Z || World_Coord[i][j][2] > Max_Z)
                                World_Coord[i][j][2] = Standard_Coord[i][j][2];
                        }
                    }
                    World_Coord[i][j][0] = (L - World_Coord[i][j][2]) * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = (L - World_Coord[i][j][2]) * (Coord[i][j].second - P[1][2]) / P[1][1];
                }
            }
    }
}

// 计算每个符号的坐标以及四周符号坐标的平均值
void cacul_coord_2(const string& img_name) {
    for (int i = 0; i < Pattern_Size; ++i)
        for (int j = 0; j < Pattern_Size; ++j) {
            Rect region = Pattern_Border[i][j];
            Coord[i][j] = make_pair(region.x - 0.5 + region.width / 2.0 + Offset_X, region.y - 0.5 + region.height / 2.0 + Offset_Y);
        }
    for (int i = 1; i < Pattern_Size - 1; ++i)
        for (int j = 1; j < Pattern_Size - 1; ++j)
            Avg_Coord[i][j] = make_pair((Coord[i - 1][j].first + Coord[i + 1][j].first + Coord[i][j - 1].first + Coord[i][j + 1].first) / 4.0,
                (Coord[i - 1][j].second + Coord[i + 1][j].second + Coord[i][j - 1].second + Coord[i][j + 1].second) / 4.0);
    if ("standard.jpg" == img_name) {
        // 标准图案
        for (int i = 0; i < Pattern_Size; ++i)
            for (int j = 0; j < Pattern_Size; ++j)
                if ((i + j) % 2 == 1) {
                    World_Coord[i][j][0] = (L - B) * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = (L - B) * (Coord[i][j].second - P[1][2]) / P[1][1];
                    World_Coord[i][j][2] = B;
                }
                else {
                    World_Coord[i][j][0] = L * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = L * (Coord[i][j].second - P[1][2]) / P[1][1];
                }
    } else {
        // 计算世界坐标
        for (int i = 1; i < Pattern_Size - 1; ++i)
            for (int j = 1; j < Pattern_Size - 1; ++j) {
                if ((i + j) % 2 == 1) {
                    // 取平均值
                    if (fabs(Coord[i][j].first - Avg_Coord[i][j].first) > EPS || fabs(Coord[i][j].second - Avg_Coord[i][j].second) > EPS) {
                        double tmp = L + B * ((P[0][2] - Avg_Coord[i][j].first) * (L - A) + P[0][0] * Standard_Coord[i][j][0])
                            / (A * (Avg_Coord[i][j].first - Coord[i][j].first));
                        World_Coord[i][j][2] = A * (tmp - B) / (A - B);
                        if (World_Coord[i][j][2] < Min_Z || World_Coord[i][j][2] > Max_Z)
                            World_Coord[i][j][2] = Standard_Coord[i][j][2];
                    }
                    World_Coord[i][j][0] = (L - World_Coord[i][j][2]) * (Avg_Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = (L - World_Coord[i][j][2]) * (Avg_Coord[i][j].second - P[1][2]) / P[1][1];
                } else {
                    // 取当前点
                    if (fabs(Coord[i][j].first - Avg_Coord[i][j].first) > EPS || fabs(Coord[i][j].second - Avg_Coord[i][j].second) > EPS) {
                        double tmp = L + B * ((P[0][2] - Coord[i][j].first) * (L - A) + P[0][0] * Standard_Coord[i][j][0])
                            / (A * (Coord[i][j].first - Avg_Coord[i][j].first));
                        World_Coord[i][j][2] = A * (tmp - B) / (A - B);
                        if (World_Coord[i][j][2] < Min_Z || World_Coord[i][j][2] > Max_Z)
                            World_Coord[i][j][2] = Standard_Coord[i][j][2];
                    }
                    World_Coord[i][j][0] = (L - World_Coord[i][j][2]) * (Coord[i][j].first - P[0][2]) / P[0][0];
                    World_Coord[i][j][1] = (L - World_Coord[i][j][2]) * (Coord[i][j].second - P[1][2]) / P[1][1];
                }
            }
    }
}

// 计算每个符号的坐标以及四周符号坐标的平均值
void cacul_coord_1(const string& img_name) {
    for (int i = 0; i < Pattern_Size; ++i)
        for (int j = 0; j < Pattern_Size; ++j) {
            Rect region = Pattern_Border[i][j];
            Coord[i][j] = make_pair((region.x - 0.5 + region.width / 2.0 + Offset_X - P[0][2]) / P[0][0],
                (region.y - 0.5 + region.height / 2.0 + Offset_Y - P[1][2]) / P[1][1]);
        }
    for (int i = 1; i < Pattern_Size - 1; ++i)
        for (int j = 1; j < Pattern_Size - 1; ++j)
            Avg_Coord[i][j] = make_pair((Coord[i - 1][j].first + Coord[i + 1][j].first + Coord[i][j - 1].first + Coord[i][j + 1].first) / 4.0,
                (Coord[i - 1][j].second + Coord[i + 1][j].second + Coord[i][j - 1].second + Coord[i][j + 1].second) / 4.0);
    if ("standard.jpg" == img_name) {
        // 标准图案
        for (int i = 0; i < Pattern_Size; ++i)
            for (int j = 0; j < Pattern_Size; ++j) {
                World_Coord[i][j][0] = L / F * Coord[i][j].first;
                World_Coord[i][j][1] = L / F * Coord[i][j].second;
            }
    } else {
        // 计算世界坐标
        for (int i = 1; i < Pattern_Size - 1; ++i)
            for (int j = 1; j < Pattern_Size - 1; ++j) {
                if ((i + j) % 2 == 1) {
                    // 取平均值
                    double son = B * (Coord[i][j].first * Standard_Coord[i][j][1] + Coord[i][j].second * Standard_Coord[i][j][0]);
                    double mon = (A - B) * (Coord[i][j].first * Avg_Coord[i][j].second - Coord[i][j].second * Avg_Coord[i][j].first);
                    World_Coord[i][j][0] = Avg_Coord[i][j].first * son / mon;
                    World_Coord[i][j][1] = -Avg_Coord[i][j].second * son / mon;
                    World_Coord[i][j][2] = L - F * son / mon;
                }
                else {
                    // 取当前点
                    double son = B * (Avg_Coord[i][j].second * Standard_Coord[i][j][0] + Avg_Coord[i][j].first * Standard_Coord[i][j][1]);
                    double mon = A * (Avg_Coord[i][j].second * Coord[i][j].first - Avg_Coord[i][j].first * Coord[i][j].second);
                    World_Coord[i][j][0] = Coord[i][j].first * son / mon;
                    World_Coord[i][j][1] = -Coord[i][j].second * son / mon;
                    World_Coord[i][j][2] = L - F * son / mon;
                }
            }
    }
}

// 将图案中心标记红色
void mark_red(const string& img_name) {
    Mat color_img = imread(Base_Dir + img_name);
    for (int i = 0; i < Pattern_Size; ++i)
        for (int j = 0; j < Pattern_Size; ++j) {
            // 中心
            pair<double, double>center = Coord[i][j];
            int y = (int)center.first, x = (int)center.second;
            color_img.at<Vec3b>(x, y) = Vec3b(0, 0, 255);
            // 边框
            Rect region = Pattern_Border[i][j];
            for (int row = region.y; row < region.y + region.height; ++row)
                color_img.at<Vec3b>(row + Offset_Y, region.x + Offset_X) = color_img.at<Vec3b>(row + Offset_Y, region.x + region.width - 1 + Offset_X) = Vec3b(0, 0, 255);
            for (int col = region.x; col < region.x + region.width; ++col)
                color_img.at<Vec3b>(region.y + Offset_Y, col + Offset_X) = color_img.at<Vec3b>(region.y + region.height - 1 + Offset_Y, col + Offset_X) = Vec3b(0, 0, 255);
            for (int row = region.y; row < region.y + region.height; ++row)
                gray_img.at<uchar>(row, region.x) = gray_img.at<uchar>(row, region.x + region.width - 1) = 255;
            for (int col = region.x; col < region.x + region.width; ++col)
                gray_img.at<uchar>(region.y, col) = gray_img.at<uchar>(region.y + region.height - 1, col) = 255;
        }
    imwrite("img/mark_red/" + img_name, color_img);
    imwrite("img/mark_red/g_" + img_name, gray_img);
}

vector<vector<vector<double>>> start(const string& img_name, bool flag=false) {
    clock_t t = clock();
    init(flag);
    // 制作13x13的模板图案
    make_template();
    // 读取灰度图和hsv图
    read_gray_hsv(img_name);
    // 剪裁两边大片无关区域，去除亮度低于40的背景像素，并二值化
    bright_above();
    // 先做闭运算，保留最大的连通域，提取粗略边界(稍微留些边缘)
    extract_border();
    // 局部均值二值化，再进行开闭运算去噪
    avg_thresh();
    // 分解成一个个连通域
    split_domain(10);
    // 寻找每一个符号
    search_pattern();
    // 标记识别错误的地方并打印结果
    mark_print(img_name);
    // 计算每个符号的坐标以及四周符号坐标的平均值，然后计算世界坐标值
    cacul_coord(img_name);
    
    mark_red(img_name);
    printf("Total time is %.3f s\n\n", (float)(clock() - t) / CLOCKS_PER_SEC);
    return World_Coord;
}

int main() {
    for (int i = 1; i <= 1; ++i) {
        start(to_string(i) + ".jpg", false);
    }
    return 0;
}

PYBIND11_MODULE(pattern, m) {
    m.doc() = "cv module";
    m.def("get_coord", &start, py::arg("img_name"), py::arg("flag") = false, "cv module");
}