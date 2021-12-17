#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 本文件只允许依赖math库
import math

# 注：参考的文章 or 内容都以列在了每个算法的开头处


def draw_line(p_list, algorithm):
    """绘制线段

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 线段的起点和终点坐标
    :param algorithm: (string) 绘制使用的算法，包括'DDA'和'Bresenham'，此处的'Naive'仅作为示例，测试时不会出现
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    x0, y0 = p_list[0]
    x1, y1 = p_list[1]
    result = []
    if algorithm == 'Naive':
        if x0 == x1:
            for y in range(y0, y1 + 1):
                result.append((int(x0), int(y)))
        else:
            if x0 > x1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            k = (y1 - y0) / (x1 - x0)
            for x in range(x0, x1 + 1):
                result.append((int(x), int(y0 + k * (x - x0))))
    elif algorithm == 'DDA':
        if x0 == x1:
            # 特殊情况的判断
            # 斜率不存在的情况
            if y0 <= y1:
                for y in range(y0, y1 + 1):
                    result.append((int(x0), int(y)))
            else:
                y = y0
                for i in range(y1, y0 + 1):
                    result.append((int(x0), int(y)))
                    y -= 1
        else:
            dx = x1 - x0
            dy = y1 - y0
            # 斜率
            k = float(dy) / dx
            if abs(dx) >= abs(dy):
                # 斜率绝对值小于等于1
                y = float(y0)
                if x1 > x0:
                    # 起始端点在左边
                    for x in range(x0, x1 + 1):
                        result.append((int(round(x)), int(round(y))))
                        y += k
                else:
                    # 起始端点在右边
                    x = x0
                    for i in range(x1, x0 + 1):
                        result.append((int(round(x)), int(round(y))))
                        x -= 1
                        y -= k
            elif abs(dx) < abs(dy):
                # 斜率大于1
                x = float(x0)
                if y1 > y0:
                    # 起始端点在左侧
                    for y in range(y0, y1 + 1):
                        result.append((int(round(x)), int(round(y))))
                        x += (1 / float(k))
                else:
                    # 起始端点在右侧
                    y = y0
                    for i in range(y1, y0 + 1):
                        result.append((int(round(x)), int(round(y))))
                        x -= (1 / float(k))
                        y -= 1
    elif algorithm == 'Bresenham':
        if x0 == x1:
            # 特殊情况的判断
            # 斜率不存在的情况
            if y1 >= y0:
                y = y0
                for i in range(int(y0), int(y1 + 1)):
                    result.append((int(x0), int(y)))
                    y += 1
            else:
                y = y0
                for i in range(int(y1), int(y0 + 1)):
                    result.append((int(x0), int(y)))
                    y -= 1
        elif y0 == y1:
            # 特殊情况的判断
            # 斜率为0的情况
            x = x0
            if x1 >= x0:
                for i in range(int(x0), int(x1 + 1)):
                    result.append((int(x), int(y0)))
                    x += 1
            else:
                x = x0
                for i in range(int(x1), int(x0 + 1)):
                    result.append((int(x), int(y0)))
                    x -= 1
        else:
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            s1 = 1 if ((x1 - x0) > 0) else -1
            s2 = 1 if ((y1 - y0) > 0) else -1

            # 初始误差
            if dx >= dy:
                # 斜率绝对值小于等于1
                result = draw_by_bresenham(dx, dy, x0, y0, s1, s2, False)
            else:
                # 斜率绝对值大于1
                # 将y当成变化维度,即交换dx与dy
                result = draw_by_bresenham(dy, dx, x0, y0, s1, s2, True)

    return result


def draw_by_bresenham(dx, dy, x0, y0, s1, s2, is_change_dimension):
    result = []
    p = 2 * dy - dx
    x = x0
    y = y0
    for i in range(0, int(dx + 1)):
        result.append((int(x), int(y)))
        if p >= 0:
            if is_change_dimension:
                x += s1
            else:
                y += s2
            p -= 2 * dx

        if is_change_dimension:
            y += s2
        else:
            x += s1
        p += 2 * dy

    return result


def draw_polygon(p_list, algorithm):
    """绘制多边形

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 多边形的顶点坐标列表
    :param algorithm: (string) 绘制使用的算法，包括'DDA'和'Bresenham'
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """
    result = []
    for i in range(len(p_list)):
        line = draw_line([p_list[i - 1], p_list[i]], algorithm)
        result += line
    return result


def draw_polygon_gui(p_list, algorithm):
    """绘制多边形

        :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 多边形的顶点坐标列表
        :param algorithm: (string) 绘制使用的算法，包括'DDA'和'Bresenham'
        :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
        """
    result = []
    for i in range(len(p_list) - 1):
        line = draw_line([p_list[i], p_list[i + 1]], algorithm)
        result += line
    return result


# https://blog.csdn.net/orbit/article/details/7496008?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242


def draw_ellipse(p_list):
    """绘制椭圆（采用中点圆生成算法）

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 椭圆的矩形包围框左上角和右下角顶点坐标
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """

    result = []
    x0, y0 = p_list[0]
    x1, y1 = p_list[1]
    # 椭圆的对称性，只需要绘制四分之一就好
    xe, ye = int((x0 + x1) / 2), int((y0 + y1) / 2)  # xe: 椭圆的上顶点横坐标, ye:椭圆的右顶点纵坐标
    a = int(abs(x1 - x0) / 2)  # 椭圆的长轴
    b = int(abs(y1 - y0) / 2)  # 椭圆的短轴，如果是圆的话两者相等

    # 决策参数
    p = b ** 2 - a ** 2 * b + a ** 2 / 4
    # 椭圆平移到原点的第一个点(0, b)
    x, y = 0, b
    # 以下两个点是确定的点，分别是椭圆的上顶点和下顶点
    result.append((x + xe, y + ye))
    result.append((x + xe, -y + ye))

    # 斜率大于一的时候，在区域1
    while b ** 2 * x < a ** 2 * y:
        x += 1
        if p < 0:
            # 选择像素点(x_k+1, y_k)
            p += 2 * b ** 2 * x + b ** 2
        else:
            # 选择像素点(x_k+1, y_k-1)
            y -= 1
            p += 2 * b ** 2 * x - 2 * a ** 2 * y + b ** 2
        # 根据对称性添加点
        result.append((x + xe, y + ye))
        result.append((-x + xe, y + ye))
        result.append((x + xe, -y + ye))
        result.append((-x + xe, -y + ye))

    # 此时到达斜率为1的时候，进入区域2
    p = b ** 2 * (x + 0.5) ** 2 + a ** 2 * (y - 1) ** 2 - a ** 2 * b ** 2
    while y >= 0:
        y -= 1
        if p > 0:
            p += -2 * a ** 2 * y + a ** 2
        else:
            x += 1
            p += 2 * b ** 2 * x - 2 * a ** 2 * y + a ** 2
        # 根据对称性添加点
        result.append((x + xe, y + ye))
        result.append((-x + xe, y + ye))
        result.append((x + xe, -y + ye))
        result.append((-x + xe, -y + ye))
    return result


# https://www.vectormoon.net/2020/09/25/Bezier/
# https://www.bilibili.com/read/cv7758777
# https://zh.wikipedia.org/wiki/%E8%B2%9D%E8%8C%B2%E6%9B%B2%E7%B7%9A


def draw_curve(p_list, algorithm):
    """绘制曲线

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 曲线的控制点坐标列表
    :param algorithm: (string) 绘制使用的算法，包括'Bezier'和'B-spline'（三次均匀B样条曲线，曲线不必经过首末控制点）
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 绘制结果的像素点坐标列表
    """

    res = []
    if algorithm == "Bezier":
        # 绘制贝塞尔曲线
        t_step = 0.0005
        t = 0
        while t <= 1:
            res.append(bezier_point(t, p_list))
            t = t + t_step
    elif algorithm == "B-spline":
        # 绘制3次均匀B样条曲线
        k = 3
        n = len(p_list) - 1  # num of control points is n+1
        u = k
        step = 0.001
        while u <= n + 1:
            p_x = 0.0
            p_y = 0.0
            for i in range(n + 1):
                nik = base_function(i, k + 1, u)
                p_x = p_x + p_list[i][0] * nik
                p_y = p_y + p_list[i][1] * nik
            u = u + step
            res.append([int(p_x), int(p_y)])
    return res


def bezier_point(t, p_list):
    """针对某个点的t值做出计算

    :param t: (float)比例
    :param p_list:(list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 曲线的控制点坐标列表
    :return: list of int:[x, y] 对应t值下Bezier曲线生成的点
    """
    result = []
    while len(p_list) > 1:
        for i in range(0, len(p_list) - 1):
            qx = (1 - t) * p_list[i][0] + t * p_list[i + 1][0]
            qy = (1 - t) * p_list[i][1] + t * p_list[i + 1][1]
            result.append([qx, qy])
        p_list = result
        result = []

    x = int(p_list[0][0])
    y = int(p_list[0][1])
    return x, y


def base_function(i, k, u):
    """ 计算B样条曲线的基函数取值

    :param i: (int) index of base function
    :param k: (int) 阶数 degree + 1
    :param u: parameter 参数
    :return:  the value of base function
    """
    nik_u = 0.0
    if k == 1:
        if i + 1 > u >= i:
            nik_u = 1.0
        else:
            nik_u = 0.0
    else:
        nik_u = ((u - i) / (k - 1)) * base_function(i, k - 1, u) + ((i + k - u) / (k - 1)) * base_function(i + 1, k - 1, u)
    return nik_u


def translate(p_list, dx, dy):
    """平移变换

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param dx: (int) 水平方向平移量
    :param dy: (int) 垂直方向平移量
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []

    for i in range(len(p_list)):
        xi = p_list[i][0]
        yi = p_list[i][1]
        result.append([int(xi + dx), int(yi + dy)])
    return result


def rotate(p_list, x, y, r):
    """旋转变换（除椭圆外）

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param x: (int) 旋转中心x坐标
    :param y: (int) 旋转中心y坐标
    :param r: (int) 顺时针旋转角度（°）
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []
    # 将角度转为弧度，用于后面的三角函数计算
    angle = math.radians(360 + r)
    for i in range(len(p_list)):
        xi = p_list[i][0]
        yi = p_list[i][1]
        change_x = x + (xi - x) * math.cos(angle) - (yi - y) * math.sin(angle)
        change_y = y + (xi - x) * math.sin(angle) + (yi - y) * math.cos(angle)
        result.append([int(change_x), int(change_y)])

    return result


def scale(p_list, x, y, s):
    """缩放变换

    :param p_list: (list of list of int: [[x0, y0], [x1, y1], [x2, y2], ...]) 图元参数
    :param x: (int) 缩放中心x坐标
    :param y: (int) 缩放中心y坐标
    :param s: (float) 缩放倍数
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1], [x_2, y_2], ...]) 变换后的图元参数
    """
    result = []
    for i in range(len(p_list)):
        xi = p_list[i][0]
        yi = p_list[i][1]
        change_x = x + (xi - x) * s
        change_y = y + (yi - y) * s
        result.append([int(change_x), int(change_y)])

    return result


# https://www.cnblogs.com/cnblog-wuran/p/9813841.html
# https://www.cnblogs.com/iamfatotaku/p/12496937.html
# https://www.geeksforgeeks.org/line-clipping-set-1-cohen-sutherland-algorithm/


# 二维平面四位二进制编码
inside_code = 0b0000
left_code = 0b0001
right_code = 0b0010
top_code = 0b1000
bottom_code = 0b0100


def clip(p_list, x_min, y_min, x_max, y_max, algorithm):
    """线段裁剪

    :param p_list: (list of list of int: [[x0, y0], [x1, y1]]) 线段的起点和终点坐标
    :param x_min: 裁剪窗口左上角x坐标
    :param y_min: 裁剪窗口左上角y坐标
    :param x_max: 裁剪窗口右下角x坐标
    :param y_max: 裁剪窗口右下角y坐标
    :param algorithm: (string) 使用的裁剪算法，包括'Cohen-Sutherland'和'Liang-Barsky'
    :return: (list of list of int: [[x_0, y_0], [x_1, y_1]]) 裁剪后线段的起点和终点坐标
    """
    # 防御式编程
    if len(p_list) != 2:
        return []
    # 线段起点
    x0, y0 = p_list[0]
    # 线段终点
    x1, y1 = p_list[1]
    res = []
    if algorithm == 'Cohen-Sutherland':
        # 对顶点位置进行编码
        start_code = cal_code(x0, y0, x_min, y_min, x_max, y_max)
        end_code = cal_code(x1, y1, x_min, y_min, x_max, y_max)
        # 如果start_code和end_code都为0，说明端点都在窗口内，那么就不用裁剪直接返回
        code = [start_code, end_code]
        res = p_list
        while code[0] | code[1] != 0:
            if code[0] & code[1] != 0:
                # 说明该线段在裁剪框外面
                return []
            for i in range(2):
                x, y = res[i]
                if code[i] == 0:
                    continue  # 说明该点是inside_code
                else:
                    if left_code & code[i] != 0:
                        # 该点在左部分，求交点
                        x = x_min
                        if x0 == x1:
                            y = y0
                        else:
                            # 根据斜率公式计算交点
                            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0)
                    elif right_code & code[i] != 0:
                        # 该点在右部分，求交点
                        x = x_max
                        if x0 == x1:
                            y = y0
                        else:
                            # 根据斜率公式计算交点
                            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
                    elif bottom_code & code[i] != 0:
                        # 该点在下部分，求交点
                        y = y_max
                        x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0)
                    elif top_code & code[i] != 0:
                        # 该点在上部分，求交点
                        y = y_min
                        x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0)
                    code[i] = cal_code(x, y, x_min, y_min, x_max, y_max)
                    res[i] = [int(x), int(y)]
    elif algorithm == 'Liang-Barsky':
        # 基本参考了ppt中给出的程序实现
        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0]
        u0, u1 = 0, 1
        for k in range(4):
            if p[k] == 0:
                if q[k] < 0:
                    return []
            else:
                u = q[k] / p[k]
                if p[k] < 0:
                    u0 = max(u0, u)
                else:
                    u1 = min(u1, u)
        if u0 > u1:
            return []
        x_0 = round(x0 + u0 * dx)
        y_0 = round(y0 + u0 * dy)
        x_1 = round(x0 + u1 * dx)
        y_1 = round(y0 + u1 * dy)
        res = [[int(x_0), int(y_0)], [int(x_1), int(y_1)]]

    return res


def cal_code(x, y, x_min, y_min, x_max, y_max):
    """计算某个点在Cohen-Sutherland算法下的编码

    :param x: 点的x坐标
    :param y: 点的y坐标
    :param x_min: 裁剪窗口左上角x坐标
    :param y_min: 裁剪窗口左上角y坐标
    :param x_max: 裁剪窗口右下角x坐标
    :param y_max: 裁剪窗口右下角y坐标
    :return: (int) 点[x, y]在该算法下的编码
    """
    code = inside_code
    if x < x_min:
        # 在左半边
        code |= left_code
    elif x > x_max:
        # 在右半边
        code |= right_code
    if y < y_min:
        # y轴是朝下的
        # 在上边
        code |= top_code
    elif y > y_max:
        # 在下边
        code |= bottom_code

    return code
