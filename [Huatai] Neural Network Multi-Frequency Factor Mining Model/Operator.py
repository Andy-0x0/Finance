from Console import Console


def print_menu():
    choice_lookup = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    while True:
        print('Select Strategy Below:')
        print(f"\b{'A. GRU [15min]':30s}\b{'B. GRU+Attention [15min]':30s}")
        print(f"\b{'C. GRU [15min+1d]':30s}\b{'D. GRU+Frozen [15min+1d]':30s}")
        choice = input()
        toggle = 0
        if choice in list('ABCDabcd'):
            choice = choice_lookup[choice.upper()]
            toggle = 1 if input('[Train & Test | Test] -> [Y | N]\n').upper() == 'Y' else 0
            break
        else:
            print('>>>[Operator]: Invalid Choice!')
            continue

    return choice, toggle


console = Console(* print_menu())
console.activate()
