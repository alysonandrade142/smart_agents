#!/usr/bin/python

import random, copy
from optparse import OptionParser

try:
    import psyco
    psyco.full()
except ImportError:
    pass

class board:
    def __init__(self, list=None):
        if list == None:
            self.board = [[0 for i in range(0, 8)] for j in range(0, 8)]
            for i in range(0, 8):
                while 1:
                    rand_row = random.randint(0, 7)
                    rand_col = random.randint(0, 7)
                    if self.board[rand_row][rand_col] == 0:
                        self.board[rand_row][rand_col] = "Q"
                        break

    def __repr__(self):
        mstr = ""
        for i in range(0, 8):
            for j in range(0, 8):
                mstr = mstr + str(self.board[i][j]) + " "
            mstr = mstr + "\n"
        return (mstr)


class queens:
    def __init__(self, numruns, verbocity, passedboard=None):
        # TODO check options
        self.totalruns = numruns
        self.totalsucc = 0
        self.totalnumsteps = 0
        self.verbocity = verbocity
        solutions = []
        for i in range(0, numruns):
            if self.verbocity == True:
                print("====================")
                print("BOARD", i)
                print("====================")
            self.mboard = board(passedboard)
            self.cost = self.calc_cost(self.mboard)
            print(self.cost)
            self.hill_solution(solutions)
            if len(solutions) >= 2:
                if solutions[i] == True & solutions[-i] == False:
                    print('Encontrou o pico', len(solutions))
                    break

    def hill_solution(self, solutions):
        while 1:
            currViolations = self.cost
            self.getlowercostboard()
            if currViolations == self.cost:
                break
            self.totalnumsteps += 1
            if self.verbocity == True:
                print("CONFLITOS/CHOQUES", self.calc_cost(self.mboard))
                print(self.mboard)
        if self.cost != 0:
            if self.verbocity == True:
                print("NENHUMA SOLUÇÃO ENCONTRADA")
                solutions.append(False)
        else:
            if self.verbocity == True:
                print("SOLUÇÃO ENCONTRADA")
                solutions.append(True)
            self.totalsucc += 1

        print('CUSTO------', self.cost)
        return self.cost

    def printstats(self):
        print("Total Percorrido: ", self.totalruns)
        print("Total Successo: ", self.totalsucc)
        #print("Porcentage de Successo: ", float(self.totalsucc) / float(self.totalruns))
        #print("Média de Passos: ", float(self.totalnumsteps) / float(self.totalruns))

    def calc_cost(self, tboard):
        totalhcost = 0
        totaldcost = 0
        for i in range(0, 8):
            for j in range(0, 8):
                if tboard.board[i][j] == "Q":
                    totalhcost -= 2
                    for k in range(0, 8):
                        if tboard.board[i][k] == "Q":
                            totalhcost += 1
                        if tboard.board[k][j] == "Q":
                            totalhcost += 1
                    k, l = i + 1, j + 1
                    while k < 8 and l < 8:
                        if tboard.board[k][l] == "Q":
                            totaldcost += 1
                        k += 1
                        l += 1
                    k, l = i + 1, j - 1
                    while k < 8 and l >= 0:
                        if tboard.board[k][l] == "Q":
                            totaldcost += 1
                        k += 1
                        l -= 1
                    k, l = i - 1, j + 1
                    while k >= 0 and l < 8:
                        if tboard.board[k][l] == "Q":
                            totaldcost += 1
                        k -= 1
                        l += 1
                    k, l = i - 1, j - 1
                    while k >= 0 and l >= 0:
                        if tboard.board[k][l] == "Q":
                            totaldcost += 1
                        k -= 1
                        l -= 1
        return ((totaldcost + totalhcost) / 2)

    def getlowercostboard(self):
        lowcost = self.calc_cost(self.mboard)
        lowestavailable = self.mboard
        for q_row in range(0, 8):
            for q_col in range(0, 8):
                if self.mboard.board[q_row][q_col] == "Q":
                    for m_row in range(0, 8):
                        for m_col in range(0, 8):
                            if self.mboard.board[m_row][m_col] != "Q":
                                tryboard = copy.deepcopy(self.mboard)
                                tryboard.board[q_row][q_col] = 0
                                tryboard.board[m_row][m_col] = "Q"
                                thiscost = self.calc_cost(tryboard)
                                if thiscost < lowcost:
                                    lowcost = thiscost
                                    lowestavailable = tryboard
        self.mboard = lowestavailable
        self.cost = lowcost


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-q", "--quiet", dest="verbose",
                      action="store_false", default=True,
                      help="Não imprima todos os movimentos")

    parser.add_option("--numrun", dest="numrun", help="Número de quadros randomicos", default=2,
                      type="int")

    (options, args) = parser.parse_args()

    mboard = queens(verbocity=options.verbose, numruns=options.numrun)
    mboard.printstats()