# Optimal Transport & Sinkhorn Algorithm (Animated Slides)


This repository presents an animated slide deck that visually explains the Discrete Optimal Transport problem and the Sinkhorn algorithm.  Built with [Manim](https://www.manim.community/) and [manim-slides](https://github.com/jeertmans/manim-slides), it walks through:

1. Formulation of the discrete Optimal Transport problem
2. Cost and transport matrices
3. Entropy regularization and the Sinkhorn algorithm
4. Heatmap visualizations of transport plans
5. Applications in machine learning (e.g., Sinkhorn distance)

Please see our [Project Proposal](project_proposal.pdf)

![Slide Preview](slides.gif)

![Sinkhorn Iterations](sinkhorn.gif)

---

## Usage

Render the slides locally:

```bash
manim render optimal_transport_slides

```

Alternatively, run with manim-slides to navigate slides interactively:

```bash
manim-slides render optimal_transport_slides
```

The resulting video or live slideshow will demonstrate each step of the algorithm with animations.

---

## Dependencies

* `manim` (v0.18 or higher)
* `manim-slides`
* `numpy`
* Python standard libs: `itertools`, `random`, `pickle`


