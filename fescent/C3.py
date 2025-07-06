import jax.numpy as jnp
import fescent
import numpy as np

import matplotlib.pyplot as plt


def penaliced_optimization(
    x0: jnp.ndarray,
    mu0: float,
    eps: float,
    beta: float,
    solver: callable,
    alpha_solver: float,
    beta_solver: float = None,
    problem: int = None,
) -> tuple[jnp.ndarray, float, int]:
    """
    Penalized optimization method for solving constrained optimization problems.
    Args:
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

    x_list = []

    while mu * actual_alpha(x) >= eps:
        if solver == fescent.optimize_nesterov:
            x, f_val = solver(x, mu, alpha_solver, beta_solver, 100, problem)
            x = np.array(x)
        else:
            x, f_val = solver(mu, alpha_solver, 100)

        mu = beta * mu
        k += 1
        f_list.append(f_val)
        x_list.append(x.copy())

    return x_list, f_list, k


def plotter(
    x_list: list[jnp.ndarray],
    f_list: list[float],
    A: jnp.ndarray,
    b: jnp.ndarray,
    E: jnp.ndarray,
    e: jnp.ndarray,
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
    xs = jnp.stack([jnp.array(x) for x in x_list])  # shape (n_iter, dim)
    fs = jnp.array(f_list)

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
        X, Y = jnp.meshgrid(
            jnp.linspace(x_min - margin_x, x_max + margin_x, 200),
            jnp.linspace(y_min - margin_y, y_max + margin_y, 200),
        )
        pts = jnp.vstack([X.ravel(), Y.ravel()])  # shape (2, N)

        # Inequality mask: A x <= b
        Ai = jnp.array(A)
        bi = jnp.array(b)
        mask_ineq = jnp.all(Ai @ pts <= bi[:, None], axis=0)

        # Equality mask: Ex = e (approx)
        Ei = jnp.array(E)
        ei = jnp.array(e)
        tol = 1e-6
        mask_eq = jnp.all(jnp.abs(Ei @ pts - ei[:, None]) < tol, axis=0)

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
            c=jnp.arange(len(xs)),
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
x0 = jnp.array([0.0, 0.0])

alpha_solver = 0.001
beta_solver = 0.9

x_list, f_list, iterations = penaliced_optimization(
    x0, mu0, eps, beta, fescent.optimize_nesterov, alpha_solver, beta_solver, 1
)

print(f"La cantidad de iteraciones es: {iterations}")

A = jnp.array([[1, 1]])
b = jnp.array([-100])

E = jnp.array([[0, 0]])
e = jnp.array([0])

plotter(
    x_list,
    f_list,
    A,
    b,
    E,
    e,
    title="Primer Problema de Optimización Penalizada con Nesterov",
)

x0 = jnp.array([0.0, 0.0])

x_list, f_list, iterations = penaliced_optimization(
    x0, mu0, eps, beta, fescent.optimize_nesterov, alpha_solver, beta_solver, 2
)

print(f"La cantidad de iteraciones es: {iterations}")

A = jnp.array([[1, 1]])
b = jnp.array([5])

E = jnp.array([[1, -5]])
e = jnp.array([2])

plotter(
    x_list,
    f_list,
    A,
    b,
    E,
    e,
    title="Segundo Problema de Optimización Penalizada con Nesterov",
)

# n = 1000

# x0 = jnp.ones(n)

# x_list, f_list, iterations = penaliced_optimization(
#     x0, mu0, eps, beta, optimize_sgd, alpha_solver
# )

# print(f"La cantidad de iteraciones es: {iterations}")

# A = jnp.ones((1, n))
# b = jnp.array([n + 1])

# E = jnp.zeros((1, n))
# e = jnp.zeros(n)

# plotter(
#     x_list,
#     f_list,
#     A,
#     b,
#     E,
#     e,
#     title="Tercer Problema de Optimización Penalizada con SGD",
# )
