# import jax
# import jax.numpy as jnp
from fescent import *
import numpy as np

import matplotlib.pyplot as plt


def alpha(
    x: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray,
    e: np.ndarray,
) -> float:

    phi = lambda y: np.maximum(0.0, y) ** 2
    psi = lambda z: np.square(z)

    term1 = phi(A @ x - b).sum()
    term2 = psi(E @ x - e).sum()
    return (term1 + term2).item()


def penaliced_optimization(
    f: callable,
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray,
    e: np.ndarray,
    x0: np.ndarray,
    mu0: float,
    eps: float,
    beta: float,
    solver: callable,
    alpha_solver: float,
    beta_solver: float = None,
    problem: int = None,
) -> tuple[np.ndarray, float, int]:
    """
    Penalized optimization method for solving constrained optimization problems.
    Args:
        f: Objective function to minimize.
        A: Coefficient matrix for the linear constraints.
        b: Right-hand side vector for the linear constraints.
        E: Coefficient matrix for the equality constraints.
        e: Right-hand side vector for the equality constraints.
        x0: Initial point for the optimization.
        mu0: Initial penalty parameter.
        eps: Convergence tolerance.
        beta: Penalty parameter increase factor.
        solver: Function to solve the optimization problem at each iteration.
        alpha_solver: Step size for the solver.
        beta_solver: Optional step size adjustment for the solver.
        problem: Tag for the problem.
    Returns:
        A tuple containing the optimal point, the value of the objective function at that point, and the number of iterations.
    """

    k = 0
    mu = mu0
    x = x0.copy()

    f_list = []
    f_list.append(f(x))

    x_list = []
    x_list.append(x.copy())

    actual_alpha = lambda x: alpha(x, A, b, E, e)

    while mu * actual_alpha(x) >= eps:
        if solver == optimize_nesterov:
            x, _, _ = solver(x * 1.0, mu, alpha_solver, beta_solver, 100, problem)
        else:
            x, _, _ = solver(x * 1.0, mu, alpha_solver, 100)

        mu = beta * mu
        alpha_solver *= 1/beta
        k += 1
        f_list.append(f(x))
        x_list.append(x.copy())

    return x_list, f_list, k


def plotter(
    x_list: list[np.ndarray],
    f_list: list[float],
    A: np.ndarray,
    b: np.ndarray,
    E: np.ndarray,
    e: np.ndarray,
    title: str = "Optimization Progress",
) -> None:
    """
    Plots f along the optimization iterations and the trajectory of x, if len(x) = 2.
    The trajectory of x is plotted with a heatmap over the iterations along with the feasible region defined by the constraints.

    We color with a pattern the feasible region defined by the constraints assuming Ax <= b and Ex = e.

    The color map has a legend.
    Args:
        x_list: List of x values at each iteration.
        f_list: List of function values at each iteration.
        A: Coefficients for inequality constraints.
        b: Right-hand side for inequality constraints.
        E: Coefficients for equality constraints.
        e: Right-hand side for equality constraints.
        title: Title of the plot.
    Returns:
        None
    """
    # Convert to numpy for plotting
    xs = np.stack([np.array(x) for x in x_list])  # shape (n_iter, dim)
    fs = np.array(f_list)

    # Prepare figure
    fig, (ax_f, ax_xy) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Plot f values
    ax_f.plot(fs, marker="o")
    ax_f.set_xlabel("Iteración")
    ax_f.set_ylabel("f(x)")
    ax_f.grid(True)

    # Only plot trajectory if 2D
    if xs.shape[1] == 2:
        # Feasible region heatmap
        # Define grid over extents of xs
        x_min, x_max = xs[:, 0].min(), xs[:, 0].max()
        y_min, y_max = xs[:, 1].min(), xs[:, 1].max()
        margin_x = 0.1 * (x_max - x_min)
        margin_y = 0.1 * (y_max - y_min)
        X, Y = np.meshgrid(
            np.linspace(x_min - margin_x, x_max + margin_x, 200),
            np.linspace(y_min - margin_y, y_max + margin_y, 200),
        )
        pts = np.vstack([X.ravel(), Y.ravel()])  # shape (2, N)

        # Inequality mask: A x <= b
        Ai = np.array(A)
        bi = np.array(b)
        mask_ineq = np.all(Ai @ pts <= bi[:, None], axis=0)

        # Equality mask: Ex = e (approx)
        Ei = np.array(E)
        ei = np.array(e)
        tol = 0.1
        mask_eq = np.all(np.abs(Ei @ pts - ei[:, None]) < tol, axis=0)

        # Combine feasible: inequality and equality
        mask = mask_ineq & mask_eq
        Z = mask.reshape(X.shape)

        # Plot feasible region
        ax_xy.contourf(
            X,
            Y,
            Z,
            levels=[-0.5, 0.5, 1.5],
            colors=["none", "lightgreen"],
            hatches=["", "////"],
            alpha=0.3,
        )

        # Plot trajectory colored by iteration
        sc = ax_xy.scatter(
            xs[:, 0],
            xs[:, 1],
            c=np.arange(len(xs)),
            cmap="viridis",
            label="Trayectoria",
        )
        # Connect the dots in trajectory
        ax_xy.plot(xs[:, 0], xs[:, 1], linestyle="-", linewidth=1)

        # Colorbar for iterations
        cbar = fig.colorbar(sc, ax=ax_xy)
        cbar.set_label("iteración")

        ax_xy.set_xlabel("x[0]")
        ax_xy.set_ylabel("x[1]")
        ax_xy.set_title("Trayectoria con Conjunto Factible")
        ax_xy.legend()
    else:
        ax_xy.text(
            0.5, 0.5, "Trayectoria disponible solo para x 2D", ha="center", va="center"
        )
        ax_xy.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"{title}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


mu0 = 1.0
beta = 2.0
eps = 1e-3
x0 = np.array([0.0, 0.0])

alpha_solver = 0.1
beta_solver = 0.1


def f(x: np.ndarray) -> float:
    return np.sum(np.square(x))


A = np.array([[1, 1]])
b = np.array([-100])

E = np.array([[0, 0]])
e = np.array([0])

x_list, f_list, iterations = penaliced_optimization(
    f, A, b, E, e, x0, mu0, eps, beta, optimize_nesterov, alpha_solver, beta_solver, 1
)

print(f"La cantidad de iteraciones para problema 1 con Nesterov: {iterations}")

plotter(
    x_list,
    f_list,
    A,
    b,
    E,
    e,
    title="Primer Problema de Optimización Penalizada con Nesterov",
)

x0 = np.array([0.0, 0.0])


def f(x: np.ndarray) -> np.ndarray:
    return (1 - x[0]) ** (3 / 2) + 100 * (x[1] - x[0] ** 2) ** 2


A = np.array([[1, 1]])
b = np.array([5])

E = np.array([[1, -5]])
e = np.array([2])

alpha_solver = 0.001


x_list, f_list, iterations = penaliced_optimization(
    f, A, b, E, e, x0, mu0, eps, beta, optimize_nesterov, alpha_solver, beta_solver, 2
)

print(f"La cantidad de iteraciones para problema 2 con Nesterov es: {iterations}")

plotter(
    x_list,
    f_list,
    A,
    b,
    E,
    e,
    title="Segundo Problema de Optimización Penalizada con Nesterov",
)

n = 1000

x0 = np.ones(n, dtype=float)


def f(x: np.ndarray) -> np.ndarray:
    x0 = x[:-1]
    x1 = x[1:]
    return np.sum(50 * (x1 - x0**2) ** 2 + (1 - x0) ** 2)


A = np.zeros((1, n))
b = np.zeros([n + 1])

E = np.ones((1, n))
e = np.array([n + 1])

alpha_solver = 1e-4

x_list, f_list, iterations = penaliced_optimization(
    f, A, b, E, e, x0, mu0, eps, beta, optimize_sgd, alpha_solver
)

print(f"La cantidad de iteraciones para el problema de SGD es: {iterations}")

plotter(
    x_list,
    f_list,
    A,
    b,
    E,
    e,
    title="Tercer Problema de Optimización Penalizada con SGD",
)
