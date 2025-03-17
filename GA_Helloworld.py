import numpy as np

def helloga(target='Hello world'):
    GA_POPSIZE = 30
    GA_MAXITER = 25003
    GA_ELITRATE = 0.1
    GA_MUTATION = 0.25
    PRINTOUT = 10

    t = 0
    target_ascii = np.array([ord(c) for c in target])
    P = np.floor(np.random.rand(GA_POPSIZE, len(target)) * 96 + 32).astype(int)
    F = np.sum(np.abs(P - target_ascii), axis=1)

    while t < GA_MAXITER and F[0] != 0:
        F, I = np.sort(F), np.argsort(F)
        P = P[I]
        y = ''.join([chr(c) for c in P[0]])

        if t % PRINTOUT == 0:
            print(f'{t}. generation best individual: {y}   fitness: {F[0]}')

        elites = int(GA_ELITRATE * GA_POPSIZE)
        Puj = np.copy(P[:elites])

        for i in range(elites, GA_POPSIZE):
            e1, e2 = np.random.randint(0, GA_POPSIZE, 2)
            crp = np.random.randint(1, len(target))
            Puj = np.vstack([Puj, np.concatenate([P[e1, :crp], P[e2, crp:]])])

        for _ in range(int(GA_POPSIZE * GA_MUTATION)):
            Puj[np.random.randint(0, GA_POPSIZE), np.random.randint(0, len(target))] = np.random.randint(32, 128)

        P = Puj
        F = np.sum(np.abs(P - target_ascii), axis=1)
        t += 1

    F, I = np.sort(F), np.argsort(F)
    y = ''.join([chr(c) for c in P[I[0]]])
    return y

# Example usage
result = helloga()
print(f'Final result: {result}')
