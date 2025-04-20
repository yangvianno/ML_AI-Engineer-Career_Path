import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Reuse the previous dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Learning rates to try
learning_rates = [0.001, 0.01, 0.1]

# Gradient functions (same as before)
def get_gradient_at_b(x, y, b, m):
    N = len(x)
    diff = np.sum(y - (m * x + b))
    return -(2 / N) * diff

def get_gradient_at_m(x, y, b, m):
    N = len(x)
    diff = np.sum(x * (y - (m * x + b)))
    return -(2 / N) * diff

# Initialize for each learning rate
steps = 30
trajectories = []

for lr in learning_rates:
    b, m = 0, 0
    traj = []
    for _ in range(steps):
        y_pred = m * x + b
        traj.append((m, b, y_pred.copy()))
        b_grad = get_gradient_at_b(x, y, b, m)
        m_grad = get_gradient_at_m(x, y, b, m)
        b -= lr * b_grad
        m -= lr * m_grad
    trajectories.append(traj)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x, y, color='blue', label='Data Points')
line, = ax.plot([], [], lw=2)
title = ax.set_title('')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(1, 6)
ax.grid(True)
ax.legend()

# Animation update function
def update(frame):
    lr_index = frame // steps
    step = frame % steps
    m, b, y_pred = trajectories[lr_index][step]
    line.set_data(x, y_pred)
    title.set_text(f"Learning Rate: {learning_rates[lr_index]} | Step: {step}")
    return line, title

# Total frames = steps per learning rate * number of learning rates
ani = animation.FuncAnimation(fig, update, frames=steps * len(learning_rates), interval=200, blit=False)

plt.show()

