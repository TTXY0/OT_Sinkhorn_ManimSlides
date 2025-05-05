# Optimal Transport & Sinkhorn Algorithm (Animated Slides)

![Slide Preview](slides.gif)

This repository presents an animated slide deck that visually explains the Discrete Optimal Transport problem and the Sinkhorn algorithm.  Built with [Manim](https://www.manim.community/) and [manim-slides](https://github.com/jeertmans/manim-slides), it walks through:

1. Formulation of the discrete Optimal Transport problem
2. Cost and transport matrices
3. Entropy regularization and the Sinkhorn algorithm
4. Heatmap visualizations of transport plans
5. Applications in machine learning (e.g., Sinkhorn distance)

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/TTXY0/BeerLawPhotonModel.git
   cd BeerLawPhotonModel
   ```
2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Render the slides locally:

```bash
# Quick (low quality) render:
manim -pql main.py DiscreteOTIntro

# High-quality render:
manim -pqh main.py DiscreteOTIntro
```

Alternatively, run with manim-slides to navigate slides interactively:

```bash
manim-slides main.py DiscreteOTIntro
```

The resulting video or live slideshow will demonstrate each step of the algorithm with animations.

---

## 📁 Repository Structure

```
BeerLawPhotonModel/
├── main.py                  # Manim script defining DiscreteOTIntro slide deck
├── assets/
│   └── preview.gif          # Animated GIF preview used in README
├── requirements.txt         # Exact Python dependencies
└── README.md                # This file
```

---

## Dependencies

* `manim` (v0.18 or higher)
* `manim-slides`
* `numpy`
* Python standard libs: `itertools`, `random`, `pickle`

All versions are pinned in `requirements.txt`.

---

## References

* Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*
* Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.*

