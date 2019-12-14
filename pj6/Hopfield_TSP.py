import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import random
import tkinter as Tkinter


class Hopfied_TSP(object):
    def __init__(self, root, iteration, width=560, height=330):
        self.root = root
        self.step = 0.5
        self.num = 36
        self.A = 1.5
        self.D = 1.0
        self.U0 = 0.02
        self.iteration = iteration
        self.width = width
        self.height = height
        self.dis = np.zeros((self.num, self.num))
        self.citys = np.array([[0.1, 0.1],[0.1, 0.2],[0.1, 0.3],[0.1, 0.4],[0.1, 0.5],[0.1, 0.6],[0.1, 0.7],[0.1, 0.8],[0.1, 0.9],[0.1, 1.0],[0.2, 1.0],[0.3, 1.0],[0.4, 1.0],[0.5, 1.0],[0.6, 1.0],[0.7, 1.0],[0.8, 1.0],[0.9, 1.0],[1.0, 1.0],[1.0, 0.9],[1.0, 0.8],[1.0, 0.7],[1.0, 0.6],[1.0, 0.5],[1.0, 0.4],[1.0, 0.3],[1.0, 0.2],[1.0, 0.1],[0.9, 0.1],[0.8, 0.1],[0.7, 0.1],[0.6, 0.1],[0.5, 0.1],[0.4, 0.1],[0.3, 0.1],[0.2, 0.1]])

        self.W = np.zeros((self.num, self.num, self.num, self.num))
        self.b = np.full((self.num, self.num), 2*self.A)
        self.U = 1 / 2 * self.U0 * np.log(self.num - 1) + (2 * (np.random.random((self.num, self.num))) - 1)
        self.V = np.zeros((self.num, self.num))
        self.du = np.zeros((self.num, self.num))

        self.best_path = []
        self.best_dis = np.inf
        self.canvas_nodes = []
        self.canvas = Tkinter.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)
        self.initCitys()
        self.initDis()
        self.title("Hopfied_TSP")


    def initCitys(self):

        minX, minY = self.citys[0, 0], self.citys[0, 1]
        maxX, maxY = minX, minY
        for city in self.citys:
            if city[0] < minX:
                minX = city[0]
            if city[1] < minY:
                minY = city[1]
            if city[0] > maxX:
                maxX = city[0]
            if city[1] > maxY:
                maxY = city[1]
        xoffset = 30
        yoffset = 30
        r = 5
        normX = (self.width - 2*xoffset) / (maxX-minX)
        normY = (self.height - 2*yoffset) / (maxY-minY)

        self.nodes = []
        self.canvas_nodes = []
        for city in self.citys:
            x = (city[0]-minX) * normX + xoffset
            y = self.height - (city[1]-minY) * normY + yoffset
            self.nodes.append((x, y))
            node = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#ff0000", outline="#000000", tag="node")
            self.canvas_nodes.append(node)

    def initDis(self):
        for i in range(self.num):
            city1 = self.citys[i]
            for j in range(self.num):
                city2 = self.citys[j]
                d = math.sqrt(pow(city1[0]-city2[0], 2)+pow(city1[1]-city2[1], 2))
                self.dis[i, j] = d

    def title(self, text):
        self.root.title(text)

    def delta(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def cal_W(self):
        for x in range(self.num):
            for i in range(self.num):
                for y in range(self.num):
                    for j in range(self.num):
                        self.W[x, i, y, j] = -self.A*(self.delta(x, y) + self.delta(i, j))-self.D*self.dis[x, y]*self.delta(j, i+1)

    def cal_du(self):
        for x in range(self.num):
            for i in range(self.num):
                self.du[x, i] = np.sum(np.sum(np.multiply(self.W[x, i, :, :], self.V)))
        self.du += self.b

    def updateU(self):
        self.U += self.du*self.step

    def updateV(self):
        self.V = 1 / 2*(1 + np.tanh(self.U / self.U0))

    def cal_E(self):
        e1 = np.sum(np.power(np.sum(self.V, axis=0) - 1, 2))
        e2 = np.sum(np.power(np.sum(self.V, axis=1) - 1, 2))
        index = [i for i in range(1, self.num)]
        index += [0]
        V_ = self.V[:, index]
        e3 = self.dis * V_
        e3 = np.sum(np.sum(np.multiply(self.V, e3)))
        e = 1/2*(self.A*(e1+e2) + self.D*e3)
        return e

    def check_path(self):
        col_max = np.max(self.V, axis=0)
        path = []
        for j in range(self.num):
            for i in range(self.num):
                if self.V[i, j] == col_max[j]:
                    path.append(i)
                    break
        if len(np.unique(path)) == self.num:
            return True, path
        else:
            return False, path

    def cal_distance(self, path):
        d = 0.0
        for i in range(len(path)-1):
            d += self.dis[path[i], path[i+1]]
        return d

    def line(self, path):
        self.canvas.delete("line")
        for i in range(-1, len(path)-1):
            p1 = self.nodes[path[i]]
            p2 = self.nodes[path[i+1]]
            self.canvas.create_line(p1, p2, fill="red", tags="line")

    def printPath(self, path):
        print("path: ")
        p = ""
        for i in range(len(path)):
            p += str(path[i]) + " -> "
            if i % 13 == 12:
                p += '\n'
        p += str(path[0])
        print(p)

    def mainloop(self):
        E = []
        iter_ = np.arange(1, self.iteration)
        self.cal_W()
        self.updateV()
        for i in range(1, self.iteration):
            self.cal_du()
            self.updateU()
            self.updateV()
            e = self.cal_E()
            E.append(e)
            flag, path = self.check_path()
            if flag:
                d = self.cal_distance(path)
                if d < self.best_dis:
                    self.best_path = path
                    self.best_dis = d
                    self.line(self.best_path)
                    self.title("Hopfied_TSP %d, Distance %lf" % (i, self.best_dis))
                    self.canvas.update()
                    print("step: %d\tbest distance: %lf\tenergy: %lf\n" % (i, self.best_dis, e))
        plt.title("Energy")
        plt.xlabel("step")
        plt.ylabel("energy")
        plt.plot(iter_, E)
        plt.show()
        if len(self.best_path) > 0:
            self.printPath(self.best_path)
        else:
            print("Couldn't found best path")
        self.root.mainloop()


if __name__ == '__main__':
    h_tsp = Hopfied_TSP(Tkinter.Tk(), 30000)
    h_tsp.mainloop()








