import cairocffi as cairo
import numpy as np
from sklearn.preprocessing import normalize
import sys


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, numpoints=100):
    def cbpoint(t):
        return ((1-t)**3)*p0 + 3*((1-t)**2)*t*p1 + 3*(t**2)*(1-t)*p2 + (t**3)*p3

    def cbpoint_deriv(t):
        return 3*((1-t)**2)*(p1-p0) + 6*(1-t)*t*(p2-p1) + 3*(t**2)*(p3-p2)

    numpoints -= (numpoints % 3) - 1
    points = np.linspace(0, 1, numpoints)
    return np.array([x for x in map(cbpoint, points)]), np.array([x for x in map(cbpoint_deriv, points)])


def coloured_bezier(ctx: cairo.Context, p0, p1, p2, p3, colors, width, detail=100, fade=None):
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    bez, bezd = cubic_bezier(p0, p1, p2, p3, numpoints=detail)
    bezd = normalize(bezd, norm='l2', axis=1)
    frac_sum = -.5

    for color, frac in colors:
        (r, g, b, a) = color
        if fade is None:
            ctx.set_source_rgba(r, g, b, a)
        else:
            ctx.set_source(fade_pattern(p0[0], p0[1], p3[0], p3[1], r, g, b, a, fade))
        ctx.set_line_width(frac*width)
        frac_sum += frac/2
        for i in range(0, bez.shape[0] - 3, 2):
            ctx.move_to(bez[i][0] - width * frac_sum * bezd[i][1], bez[i][1] + width * frac_sum * bezd[i][0])
            ctx.curve_to(bez[i + 1][0] - width * frac_sum * bezd[i + 1][1], bez[i + 1][1] + width * frac_sum * bezd[i + 1][0],
                         bez[i + 2][0] - width * frac_sum * bezd[i + 2][1], bez[i + 2][1] + width * frac_sum * bezd[i + 2][0],
                         bez[i + 3][0] - width * frac_sum * bezd[i + 3][1], bez[i + 3][1] + width * frac_sum * bezd[i + 3][0])

        ctx.stroke()
        frac_sum += frac/2


def fade_pattern(x0, y0, x1, y1, r, g, b, a, fade):
    pat = cairo.patterns.LinearGradient(x0, y0, x1, y1)
    if fade == 'None':
        return cairo.patterns.SolidPattern(r, g, b, a)
    elif fade == 'in':
        pat.add_color_stop_rgba(0, r, g, b, 0)
        pat.add_color_stop_rgba(1, r, g, b, a)
    elif fade == 'out':
        pat.add_color_stop_rgba(0, r, g, b, a)
        pat.add_color_stop_rgba(1, r, g, b, 0)
    elif fade == 'both':
        pat.add_color_stop_rgba(0, r, g, b, 0)
        pat.add_color_stop_rgba(0.5, r, g, b, a)
        pat.add_color_stop_rgba(1, r, g, b, 0)
    else:
        sys.stderr.write(f"Unknown fade style {fade}!\n")
    return pat