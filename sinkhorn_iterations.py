import numpy as np
from manim import *
from manim_slides import Slide
import itertools
import random

class DiscreteOTIntro(Slide):
    def construct(self):
        config.pixel_width = 2560
        config.pixel_height = 1600
        np.random.seed(70)
        self.camera.frame_width = 15
        self.camera.frame_height = 10
        n = 25

        
        source_hist_data = np.ones(n) / n 
        source_hist = BarChart(
            values=source_hist_data,
            bar_width=0.5,
            bar_fill_opacity=0.7,
            bar_stroke_width=4,
            bar_colors=['#003f5c'],
            x_length=5,
            y_length=3,
            y_range=[0, 0.2, 1],
            y_axis_config={"stroke_width": 0},
        ).shift(LEFT * 4 + UP*3.7)

        x_vals = np.linspace(-4, 4, n)
        gaussian_pdf = np.exp(-0.5 * x_vals**2)
        target_hist_data = gaussian_pdf / np.sum(gaussian_pdf)
        target_hist = BarChart(
            values=target_hist_data,
            bar_width=0.5,
            bar_fill_opacity=0.7,
            bar_stroke_width=4,
            bar_colors=['#ff6361'],
            x_length=5,
            y_length=3,
            y_range=[0, 0.2, 1],
            y_axis_config={"stroke_width": 0},
        ).shift(RIGHT * 4 + UP*3.7)


        arrow = Arrow(
            source_hist.get_right() + 0.5 * RIGHT,
            target_hist.get_left() - 0.5 * RIGHT,
            buff=0
        ).set_color(YELLOW).shift(DOWN*1)
        letter_T = Tex("T").next_to(arrow, UP)

        print("RENDERING SINKHORN ANIMATION")
        # # ANIMATION 160
        # plane = NumberPlane(
        #     x_range=[-10, 10, 1],
        #     y_range=[-7, 7, 1],
        #     background_line_style={
        #         "stroke_color": BLUE,
        #         "stroke_width": 1,
        #         "stroke_opacity": 0.5,
        #     }
        # )
        # self.add(plane)
        
        #segment_title = Tex("Sinkhorn Algorithm - Visual Intuition").scale(1).shift(UP * 4 + LEFT)

        
        # img = ImageMobject("python_sinkhorn.png")  # supports .png, .jpg, etc.
        # img.scale(1.5)  # optional scaling
        # img.to_edge(RIGHT).shift(RIGHT*1.5)  # position it

        # self.add(img)
        #self.wait()
        
        self.play(#Write(segment_title),
            Create(source_hist),
            FadeIn(arrow), FadeIn(letter_T),
            Create(target_hist),
            Create(Tex("-4").scale(0.5).next_to(source_hist, LEFT + DOWN*1.65, buff=0.1)),
            Create(Tex("0").scale(0.5).next_to(source_hist, RIGHT + DOWN*1.65, buff=0.1)),
            Create(Tex("0").scale(0.5).next_to(target_hist, LEFT + DOWN*1.65, buff=0.1)),
            Create(Tex("4").scale(0.5).next_to(target_hist, RIGHT + DOWN*1.65, buff=0.1))
            )
        
        
        self.next_slide()
        

        sinkhorn_eq = [
            [Tex(r"\textbf{Sinkhorn Algorithm}")],
            [MathTex(r"K = \exp\left(-\frac{C}{\gamma}\right)")],
            [MathTex(r"u^{(0)} = \vec{1},\quad v^{(0)} = \vec{1}")],
            [Tex(r"\text{Repeat:}")],
            [MathTex(r"v^{(k+1)} = \frac{b}{K^\top u^{(k)}}")],
            [MathTex(r"u^{(k+1)} = \frac{a}{K v^{(k+1)}}")],
            [MathTex(r"T^{(k)} = \mathrm{diag}(u^{(k)}) \, K \, \mathrm{diag}(v^{(k)})")]
        ]

        lines = len(sinkhorn_eq)
        x_offset = -5.5  
        y_offset = 1.5
        line_spacing = 0.8

        for i in range(lines):
            for j in range(len(sinkhorn_eq[i])):
                obj = sinkhorn_eq[i][j]
                obj.scale(0.5).move_to([x_offset, y_offset - i * line_spacing, 0])
                self.add(obj)

        
        ######


        def sinkhorn_iterates(a, b, C, gamma, n_iters):
            K = np.exp(-C / gamma)
            u = np.ones_like(a)
            v = np.ones_like(b)
            T_steps = []

            for _ in range(n_iters):
                u = a / (K @ v)
                v = b / (K.T @ u)
                T = np.diag(u) @ K @ np.diag(v)
                T_steps.append(T)
            return T_steps
        
        
        
        def granular_sinkhorn_iterates(a, b, C, gamma, n_iters):
            K = np.exp(-C / gamma)
            u = np.ones_like(a)
            v = np.ones_like(b)
            T_steps = []

            n = len(a)

            for it in range(n_iters):
                #  Column scaling (one column at a time) 
                for j in range(n):
                    denom = np.dot(K[:, j], u)
                    v[j] = b[j] / denom
                    T = np.diag(u) @ K @ np.diag(v)
                    T_steps.append(T.copy())

                #  Row scaling (one row at a time) 
                for i in range(n):
                    denom = np.dot(K[i, :], v)
                    u[i] = a[i] / denom
                    T = np.diag(u) @ K @ np.diag(v)
                    T_steps.append(T.copy())

                print(f"Iteration {it+1}: max T = {T.max():.4f}, min T = {T.min():.2e}")

            return T_steps

        
        
        n = 25
        x_vals = np.linspace(-4, 4, n)
        a = np.ones(n) / n
        gaussian_pdf = np.exp(-0.5 * x_vals**2)
        b = gaussian_pdf / np.sum(gaussian_pdf)
        
        x_vals_src = np.linspace(-4, 0, n)
        x_vals_tgt = np.linspace(0, 4, n)
        X, Y = np.meshgrid(x_vals_src, x_vals_tgt)
        C = (X - Y)**2
        
        gamma = 0.5   
        iterations = 25
        #T_steps = sinkhorn_iterates(a, b, C, gamma, n_iters=iterations)
        T_steps = granular_sinkhorn_iterates(a, b, C, gamma=0.1, n_iters=25)


        square_size = 0.2
        origin = [-2.5, 1, 0]

        self.next_slide()
        
        shift = 1

        print("iterating sinkhorns....")
        for k, T in enumerate(T_steps):
            print("iteration: ", k)
            local_max = T.max()
            squares = VGroup()

            for i in range(n):
                for j in range(n):
                    value = T[i, j]
                    scaled_val = value / local_max
                    color = interpolate_color(BLACK, WHITE, scaled_val)
                    square = Square(
                        side_length=square_size,
                        fill_opacity=1,
                        fill_color=color,
                        stroke_width=0
                    ).move_to(origin + RIGHT * j * square_size + DOWN * i * square_size)
                    squares.add(square)

            # step_type = "column" if k % 2 == 0 else "row"
            # step_label = MathTex(rf"T^* \text{{ after Sinkhorn Iteration }} {k/2:.1f} \text{{ ({step_type} scaling)}}")\
            #     .scale(0.6).move_to(origin + UP * 0.3 + RIGHT * 3.2)
            
            
            step_size = 1 / (2 * n)
            step_number = k * step_size

            step_type = "column" if k < n or (k % (2 * n)) < n else "row"  # optional refinement
            step_label = MathTex(
                rf"T^* \text{{ after Sinkhorn Iteration }} {step_number:.1f} \text{{ ({step_type} scaling)}}"
            ).scale(0.6).move_to(origin + UP * 0.3 + RIGHT * 2.2)

            

            self.add(squares, step_label)
            self.wait(0.05)
            #self.next_slide()
            
            if k == len(T_steps) - 1:
                self.wait()
            else:
                self.remove(squares, step_label)

