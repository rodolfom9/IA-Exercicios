# Importe as bibliotecas necessárias
from collections import deque

# Defina as dimensões do quebra-cabeça
N = 3

# Estrutura para armazenar um estado do quebra-cabeça
class PuzzleState:
    def __init__(self, board, x, y, depth):
        self.board = board
        self.x = x
        self.y = y
        self.depth = depth

# Movimentos possíveis: Esquerda, Direita, Cima, Baixo
row = [0, 0, -1, 1]
col = [-1, 1, 0, 0]

# Função para verificar se um determinado estado é o estado objetivo
def is_goal_state(board):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    return board == goal

# Função para verificar se um movimento é válido
def is_valid(x, y):
    return 0 <= x < N and 0 <= y < N

# Função para imprimir o tabuleiro
def print_board(board):
    for row in board:
        print(' '.join(map(str, row)))
    print("--------")

# Busca em Profundidade para resolver o problema do quebra-cabeça 8
def solve_puzzle_dfs(start, x, y):
    stack = []
    visited = set()

    stack.append(PuzzleState(start, x, y, 0))
    visited.add(tuple(map(tuple, start)))

    while stack:
        curr = stack.pop()

        # Imprime o tabuleiro atual
        print(f'Profundidade: {curr.depth}')
        print_board(curr.board)

        # Verifica se o estado objetivo foi alcançado
        if is_goal_state(curr.board):
            print(f'Estado objetivo alcançado na profundidade {curr.depth}')
            return

        # Explora os movimentos possíveis
        for i in range(4):
            new_x = curr.x + row[i]
            new_y = curr.y + col[i]

            if is_valid(new_x, new_y):
                new_board = [row[:] for row in curr.board]
                # Troca as peças
                new_board[curr.x][curr.y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[curr.x][curr.y]

                # Se este estado ainda não foi visitado, adiciona à pilha
                board_tuple = tuple(map(tuple, new_board))
                if board_tuple not in visited:
                    visited.add(board_tuple)
                    stack.append(PuzzleState(new_board, new_x, new_y, curr.depth + 1))

    print('Nenhuma solução encontrada (Busca em Profundidade atingiu o limite de profundidade)')

# Código principal
if __name__ == '__main__':
    start = [[0, 8, 7], [6, 5, 4], [3, 2, 1]]
    x, y = 1, 1

    print('Estado Inicial:')
    print_board(start)

    solve_puzzle_dfs(start, x, y)
    #