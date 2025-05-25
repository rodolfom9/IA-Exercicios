
def torre_hanoi_recursiva(n_discos, pino_origem, pino_destino, pino_auxiliar, lista_movimentos):
    if n_discos == 1:
        lista_movimentos.append((pino_origem, pino_destino, 1))
        return

    torre_hanoi_recursiva(n_discos - 1, pino_origem, pino_auxiliar, pino_destino, lista_movimentos)
    lista_movimentos.append((pino_origem, pino_destino, n_discos))
    torre_hanoi_recursiva(n_discos - 1, pino_auxiliar, pino_destino, pino_origem, lista_movimentos)


def imprimir_movimentos(movimentos):
    print("\nSequência de Movimentos Necessários:")
    for i, (origem, destino, disco) in enumerate(movimentos, 1):
        print(f"{i}. Mover disco do pino {origem} para o pino {destino}")
    print("-" * 30)
    print(f"Total de movimentos: {len(movimentos)}")


def main():
    numero_de_discos = 3  # Você pode alterar este valor
    movimentos_realizados = []

    print(f"Resolvendo a Torre de Hanói para {numero_de_discos} discos:")
    print("Pino de Origem: A")
    print("Pino de Destino: C")
    print("Pino Auxiliar: B")
    print("-" * 30)

    torre_hanoi_recursiva(numero_de_discos, 'A', 'C', 'B', movimentos_realizados)
    imprimir_movimentos(movimentos_realizados)
    print(f"Total de movimentos esperado: {2**numero_de_discos - 1}")


if __name__ == '__main__':
    main()