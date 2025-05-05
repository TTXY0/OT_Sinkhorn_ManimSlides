import numpy as np
from manim import *
from manim_slides import Slide
import itertools
import random
import pickle

def matrix_to_latex_C(M, name):
    lines = [f"{name} = \\begin{{pmatrix}}"]
    for row in M:
        row_str = " & ".join(f"{{val:.1f}}".format(val=val) for val in row)
        lines.append(row_str + r"\\")
    lines.append(r"\end{pmatrix}")
    return "\n".join(lines)

def matrix_to_latex_T(M, name):
    lines = [f"{name} = \\begin{{pmatrix}}"]
    for row in M:
        row_str = " & ".join(str(int(val)) for val in row)
        lines.append(row_str + r"\\")
    lines.append(r"\end{pmatrix}")
    return "\n".join(lines)

class DiscreteOTIntro(Slide):
    def construct(self):
        config.pixel_width = 2560
        config.pixel_height = 1600
        np.random.seed(70)
        self.camera.frame_width = 15
        self.camera.frame_height = 10


        title = Tex("Optimal Transport and Sinkhorn Algorithm").scale(1.2)
        sub_title = Tex("Thomas Wynn and Austine Do").scale(1).next_to(title, DOWN, buff=0.5)
        self.play(Write(title), Write(sub_title))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(sub_title))
        
        content_1 = Tex("1. Optimal Transport Problem").scale(1.2).shift(UP*3)
        conent_2 = Tex("2. Entropy Regularization and Sinkhorn").scale(1.2).next_to(content_1, DOWN, buff=2)
        content_3 = Tex("3. Applications in Machine Learning").scale(1.2).next_to(conent_2, DOWN, buff=2)
        
        self.play(Write(content_1),
                  Write(conent_2),
                  Write(content_3))
        
        self.next_slide()
        self.clear()
        
        
        line_source = NumberLine(
            x_range=[1, 3, 1],   
            length=4,
            include_numbers=True,  
            include_ticks=True,
            color=WHITE,
        ).shift(LEFT * 4.25)

        line_target = NumberLine(
            x_range=[4, 6, 1],  
            length=4,
            include_numbers=True,
            include_ticks=True,
            color=WHITE,
        ).shift(LEFT * 4.25 + RIGHT * 8.5)

        arrow = Arrow(
            line_source.get_right() + 1 * RIGHT,
            line_target.get_left() - 1 * RIGHT,
            buff=0
        ).set_color(YELLOW).shift(UP * 0.2)

        self.play(Create(line_source), Create(line_target), Create(arrow))
        #self.next_slide()

        n = 3
       
        source_positions = np.arange(1, n+1)
    
        radii_raw = [0.2, 0.5, 0.3]
        sum_r = np.sum(radii_raw)
        radii = radii_raw / sum_r

        left_circles = VGroup()
        source_coords = []
        for i in range(n):
            source_coords.append(source_positions[i])
            source_point = line_source.n2p(source_positions[i])
            circle = Circle(radius=radii_raw[i], color=BLUE, fill_opacity=0.5)
            circle.move_to(source_point)
            label = MathTex(f"{radii[i]}").set(font_size=20)
            label.next_to(circle, UP, buff=0.1)
            c_and_label = VGroup(circle, label)
            left_circles.add(c_and_label)

        self.play(LaggedStartMap(Create, left_circles, lag_ratio=0.1))
        self.next_slide()
        self.next_slide(loop=True)



        
        target_positions = np.arange(4, 7)
        np.random.shuffle(target_positions)

        moves = []
        for i, c_and_label in enumerate(left_circles):
          
            target_point = line_target.n2p(target_positions[i])
            moves.append(c_and_label.animate.move_to(target_point + [0,0.1,0]).set_color(ORANGE))

        self.play(*moves)
        self.pause()
        self.next_slide()



        T_label = Tex("T").next_to(arrow, UP)
        self.play(Write(T_label))
        self.next_slide()

        shift = 2
       
        self.play(
            line_source.animate.shift(UP*shift),
            line_target.animate.shift(UP*shift),
            arrow.animate.shift(UP*shift),
            left_circles.animate.shift(UP*shift),
            T_label.animate.shift(UP*shift)
        )



        
        cost_matrix = MathTex(
            # r"""
            # C = \begin{pmatrix}
            # c_{1,1} & c_{1,2} & \cdots & c_{1,m} \\
            # c_{2,1} & c_{2,2} & \cdots & c_{2,m} \\
            # \vdots  & \vdots  & \ddots & \vdots  \\
            # c_{n,1} & c_{n,2} & \cdots & c_{n,m}
            # \end{pmatrix}
            # """
            r"""
            C = \begin{pmatrix}
            c_{1,1} & c_{1,2} & c_{1,3} \\
            c_{2,1} & c_{2,2} & c_{2,3} \\
            c_{3,1} & c_{3,2} & c_{3,3}
            \end{pmatrix}
            """
        ).scale(0.7).shift(LEFT*3, DOWN*1.7)

        # transport_matrix = MathTex(
        #     r"""
        #     T = \begin{pmatrix}
        #     t_{1,1} & t_{1,2} & \cdots & t_{1,m} \\
        #     t_{2,1} & t_{2,2} & \cdots & t_{2,m} \\
        #     \vdots  & \vdots  & \ddots & \vdots  \\
        #     t_{n,1} & t_{n,2} & \cdots & t_{n,m}
        #     \end{pmatrix}
        #     """
        # )
        transport_matrix = MathTex(
            r"""
            T = \begin{pmatrix}
            t_{1,1} & t_{1,2} & t_{1,3} \\
            t_{2,1} & t_{2,2} & t_{2,3} \\
            t_{3,1} & t_{3,2} & t_{3,3}
            \end{pmatrix}
            """
        ).scale(0.7).next_to(cost_matrix, RIGHT, buff=1.5)
        T_label = Tex("Transport Matrix").next_to(transport_matrix, UP*1)
        C_label = MathTex(
            r"\begin{array}{c} \text{Cost Matrix} \\ \text{(Mass} \cdot \text{Distance)} \end{array}"
        ).next_to(cost_matrix, UP * 1)
        
        left_circles_initial = VGroup()
        for i in range(n):

            source_point = line_source.n2p(source_coords[i])
            circle = Circle(radius=radii_raw[i], color=BLUE, fill_opacity=0.5)
            circle.move_to(source_point)
            label = MathTex(f"{radii[i]}").set(font_size=20)
            label.next_to(circle, UP, buff=0.1)
            c_and_label = VGroup(circle, label)
            left_circles_initial.add(c_and_label)
        
        self.play(FadeIn(cost_matrix), FadeIn(transport_matrix), Write(T_label), Write(C_label))
        self.play(FadeIn(left_circles_initial, lag_ratio=0.1))
        self.next_slide()
        
        
        equation = MathTex(r"T^*=\underset{T}{argmin} \sum_{ij} C_{ij} T_{ij}").shift(UP*0.8)
        self.play(Write(equation),
                  FadeOut(left_circles))
        self.pause()
        self.next_slide()
        
        
        moves = []
        for i, c_and_label in enumerate(left_circles):
            target_point = line_target.n2p(0) + [8, i-1, 0]
            moves.append(c_and_label.animate.move_to(target_point))


        self.play(
            *moves
        )
        
        
        
        
        original_source_positions = np.arange(1, n+1)
        original_target_positions = np.arange(4, 7)
        cost_values = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                dist = abs(original_target_positions[j]- original_source_positions[i])
                #print("radii[i]:", radii[i])
                cost_values[i,j] = radii[i] * dist
        T_values = np.zeros((n,n))
        for i in range(n):
            j_min = np.argmin(cost_values[i,:])
            T_values[i,j_min] = 1.0

        cost_latex = matrix_to_latex_C(cost_values, "C")
        T_latex = matrix_to_latex_T(T_values, "T")

        cost_matrix_numeric = MathTex(cost_latex).scale(0.7).move_to(cost_matrix)
        transport_matrix_numeric = MathTex(T_latex).scale(0.7).move_to(transport_matrix).shift(RIGHT*0.5)



        self.play(
            Transform(cost_matrix, cost_matrix_numeric),
            Transform(transport_matrix, transport_matrix_numeric)
        )
        
        self.play(FadeOut(cost_matrix), FadeOut(C_label))
        
        self.pause()
        self.next_slide()

        marginal_constraint_row = MathTex(r"\text{source constraint:} \ \sum_j T_{i,j} = 1").scale(0.7).next_to(transport_matrix, LEFT*2, buff=1.5)
        marginal_constraint_column = MathTex(r"\text{target constraint:} \ \sum_i T_{i,j} = 1").scale(0.7).next_to(marginal_constraint_row, DOWN, buff=0.1)
        self.play(Write(marginal_constraint_row), Write(marginal_constraint_column), equation.animate.next_to(marginal_constraint_row, UP, buff=0.1))
        self.next_slide()
        self.next_slide(loop = True)

        target_positions = np.arange(4, 4 + n)
        random_permutations = list(itertools.permutations(target_positions))

        T_values = np.zeros((n, n))

        for _ in range(6):  
            
            T_values = np.zeros((n, n))

            # Select a random permutation
            perm = random.choice(random_permutations)

            
            for i in range(n):
                T_values[i, perm[i] - 4] = 1  # Set transport matrix 
                #print(perm[i] - 4)

            T_latex = matrix_to_latex_T(T_values, "T")
            transport_matrix_numeric = MathTex(T_latex).scale(0.7).move_to(transport_matrix_numeric)

            # Move circles
            moves = []
            for i, c_and_label in enumerate(left_circles):
                target_point = line_target.n2p(perm[i])
                moves.append(c_and_label.animate.move_to(target_point + [0, 0.1, 0]))

            self.play(
                Transform(transport_matrix, transport_matrix_numeric),
                *moves
            )

        #self.wait()
        self.next_slide()
        
        
        # ###################################
        # ###################################
        
        #             # ENTROPY REGULARIZATION AND SINKHORN
        
        # ###################################
        # ###################################
        # ANIMATION 25
        
        
        self.clear()
        title = Tex("Entropy Regularization and Sinkhorn").scale(1.2)
        self.play(Write(title))
        self.next_slide()
        
        
        print("RENDERING PART 2")
        self.clear()
        # title = Tex("Optimal Transport and Sinkhorn Algorithm").scale(1.2)
        # sub_title = Tex("Thomas Wynn and Austine Do").scale(1).next_to(title, DOWN, buff=0.5)
        # self.play(Write(title), Write(sub_title))

        
        
        # plane = NumberPlane(
        #     x_range=[-10, 10, 1],
        #     y_range=[-7, 7, 1],
        #     background_line_style={
        #         "stroke_color": BLUE,
        #         "stroke_width": 1,
        #         "stroke_opacity": 0.5,
        #     }
        # )
        # # Add the grid to the scene
        # self.add(plane)

        title = Tex("Entropy Regularization").scale(1).shift(UP*4.2, LEFT * 4.5)
        self.play(FadeIn(title))

        # Transport Matrix equation
        transport_matrix = MathTex(r"T^*=\underset{T}{argmin} \sum_{ij} C_{ij} T_{ij}")\
            .shift(DOWN*2, LEFT*4).scale(0.5)

        # Original Marginal Constraints (Summation form)
        marginal_constraint_row = MathTex(r"\text{source constraint: } \sum_j T_{i,j} = 1")\
            .scale(0.4).next_to(transport_matrix, DOWN, buff=0.1)
        marginal_constraint_column = MathTex(r"\text{target constraint: } \sum_i T_{i,j} = 1")\
            .scale(0.4).next_to(marginal_constraint_row, DOWN, buff=0.3)

        # Display original constraints and transport matrix
        self.play(FadeIn(transport_matrix))
        self.play(FadeIn(marginal_constraint_row), FadeIn(marginal_constraint_column))

    
        n = 25
        # --- Source histogram: Uniform distribution ---
        # source_values = np.random.uniform(-4, 4, 1000)
        # source_hist_data, _ = np.histogram(source_values, bins=n, range=(-4, 4))
        # source_hist_data = source_hist_data / np.sum(source_hist_data)
        
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
        ).shift(LEFT * 4)
        
        # --- Target histogram: Gaussian distribution ---
        # target_values = np.random.normal(0, 1, 1000)
        # target_hist_data, _ = np.histogram(target_values, bins=n, range=(-4, 4))
        # target_hist_data = target_hist_data / np.sum(target_hist_data)
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
        ).shift(RIGHT * 4)
        
        
        self.remove(source_hist.y_axis)
        self.remove(target_hist.y_axis)
        
        arrow = Arrow(source_hist.get_right(), target_hist.get_left(), buff=0.1)
        letter_T = Tex("T").next_to(arrow, UP)

        labels = VGroup(
            Tex("-4").scale(0.5).next_to(source_hist, LEFT + DOWN * 1.65, buff=0.1),
            Tex("0").scale(0.5).next_to(source_hist, RIGHT + DOWN * 1.65, buff=0.1),
            Tex("0").scale(0.5).next_to(target_hist, LEFT + DOWN * 1.65, buff=0.1),
            Tex("4").scale(0.5).next_to(target_hist, RIGHT + DOWN * 1.65, buff=0.1)
        )

        self.play(
            Create(source_hist),
            FadeIn(arrow), FadeIn(letter_T),
            Create(target_hist),
            *[FadeIn(label) for label in labels]
        )
        self.pause()



        ###### MOVE UP #######


        self.play(
            target_hist.animate.scale(1).shift(UP * 3.5),
            source_hist.animate.scale(1).shift(UP * 3.5),
            labels.animate.scale(1).shift(UP * 3.5),
            letter_T.animate.shift(UP * 2),
            arrow.animate.shift(UP * 2),
            transport_matrix.animate.scale(1.5).shift(UP * 3),
            marginal_constraint_column.animate.scale(1.5).shift(UP * 3),
            marginal_constraint_row.animate.scale(1.5).shift(UP * 3)
        )
        self.pause()
        self.next_slide()


        new_constraint_a =MathTex(r"\text{source constraint: } \sum_j T_{i,j} = a_{i}")\
            .scale(0.7).next_to(transport_matrix, DOWN, buff=0.1)
        new_constraint_b = MathTex(r"\text{target constraint: } \sum_i T_{i,j} = b_{j}")\
            .scale(0.7).next_to(new_constraint_a, DOWN, buff=0.3)

        self.play(
            ReplacementTransform(marginal_constraint_row, new_constraint_a),
            ReplacementTransform(marginal_constraint_column, new_constraint_b),
        )
        self.play(new_constraint_b.animate.shift(RIGHT*8, UP*1.1))
        self.pause()
        self.next_slide()
        
        matrix_constraint_a = MathTex(r"\text{source constraint: } T \mathbf{1}_n = \vec{a}")
        
        matrix_constraint_b = MathTex(r"\text{target constraint: } T^T \mathbf{1}_m = \vec{b}")
        
        matrix_constraint_a.move_to(new_constraint_a.get_center())
        matrix_constraint_b.move_to(new_constraint_b.get_center())
        
        self.play(
            ReplacementTransform(new_constraint_a, matrix_constraint_a),
            ReplacementTransform(new_constraint_b, matrix_constraint_b),
        )
        self.next_slide()
        # Add labels a_1, ..., a_n above source bars
        source_labels = VGroup()
        for i, bar in enumerate(source_hist.bars):
            if i % 2 == 0:
                label = MathTex(f"a_{{{i+1}}}", font_size=20).next_to(bar, UP, buff=0.1)
                source_labels.add(label)
        self.play(*[FadeIn(label) for label in source_labels])
                
                
        target_labels = VGroup()
        for i, bar in enumerate(target_hist.bars):
            if i % 2 == 0:
                label = MathTex(f"b_{{{i+1}}}", font_size=20).next_to(bar, UP, buff=0.1)
                target_labels.add(label)
        self.play(*[FadeIn(label) for label in target_labels])
        
        vector_a = MathTex(r"\text{source constraint: } T \mathbf{1}_n = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix}")
        vector_b = MathTex(r"\text{target constraint: } T^T \mathbf{1}_m = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}")
        vector_a.move_to(new_constraint_a.get_center())
        vector_b.move_to(new_constraint_b.get_center())
        self.play(
            ReplacementTransform(matrix_constraint_a, vector_a),
            ReplacementTransform(matrix_constraint_b, vector_b),
        )

        self.next_slide()


        matrix_constraint_a = MathTex(r"T \mathbf{1}_n = \vec{a}")
        matrix_constraint_b = MathTex(r"T^T \mathbf{1}_m = \vec{b}")
        matrix_constraint_a.move_to(vector_a.get_center())
        matrix_constraint_b.move_to(vector_b.get_center())
        
        self.play(FadeOut(source_labels), 
                FadeOut(target_labels),
                ReplacementTransform(vector_a, matrix_constraint_a),
                ReplacementTransform(vector_b, matrix_constraint_b))
        self.play(matrix_constraint_a.animate.scale(0.7).shift(UP*1, RIGHT*4),
                  matrix_constraint_b.animate.scale(0.7).shift(UP*1, LEFT*2),)
        self.next_slide()
        
        entropy_regularized = MathTex(
            r"T_\gamma^* = \underset{T}{\arg\min} \sum_{i,j} C_{ij} T_{ij} - \gamma H(T)"
        ).scale(0.7).move_to(transport_matrix.get_center())
        
        #ER_title = Tex("Sinkhorn Algorithm - Entropy Regularization").scale(1).shift(UP*4.2, LEFT * 2)
        self.play(ReplacementTransform(transport_matrix, entropy_regularized))

        
        
        # row-by-row mass transport using existing histograms
        self.pause()
        gamma_for_demo = 0.5

        n = 25
        x_vals = np.linspace(-4, 4, n)
        a = np.ones(n) / n
        gaussian_pdf = np.exp(-0.5 * x_vals**2)
        b = gaussian_pdf / np.sum(gaussian_pdf)
        
        x_vals_src = np.linspace(-4, 0, n)
        x_vals_tgt = np.linspace(0, 4, n)
        X, Y = np.meshgrid(x_vals_src, x_vals_tgt)
        C = (X - Y)**2

        def sinkhorn(a, b, C, gamma, num_iters=100):
            K = np.exp(-C / gamma)
            u = np.ones_like(a)
            v = np.ones_like(b)
            for _ in range(num_iters):
                u = a / (K @ v)
                v = b / (K.T @ u)
            T = np.diag(u) @ K @ np.diag(v)
            return T

        T_demo = sinkhorn(a, b, C, gamma_for_demo)
        max_val = np.max(T_demo)

        # row_transport_title = Tex(f"Mass Transport Row-by-Row ($\\gamma = {gamma_for_demo}$)")\
        #     .scale(0.6).to_edge(UP)

        #self.play(FadeIn(row_transport_title))
        self.next_slide()
        self.play(entropy_regularized.animate.shift(LEFT*0.5),
                  FadeOut(matrix_constraint_a),
                  FadeOut(matrix_constraint_b),
                  FadeOut(letter_T)
        )
        
                # Show T^* as a matrix
       # === Create heatmap of T_demo ===
        heatmap = VGroup()
        heatmap_cell_size = 0.15
        heatmap_origin = [-1.7, 1 ,0]#UR + DOWN * 0.5  # position top-left

        for i in range(n):
            for j in range(n):
                val = T_demo[i, j]
                color = interpolate_color(BLACK, WHITE, val / T_demo.max())
                cell = Square(
                    side_length=heatmap_cell_size,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0
                )
                cell.move_to(
                    heatmap_origin
                    + RIGHT * j * heatmap_cell_size
                    + DOWN * i * heatmap_cell_size
                )
                heatmap.add(cell)

        T_label = MathTex(rf"T^*, { { }} \gamma = {gamma_for_demo}").scale(0.7).next_to(heatmap, UP)
        self.play(FadeIn(T_label), FadeIn(heatmap), run_time=0.5)
        
        
        # === Create heatmap of C ===
        cost_heatmap = VGroup()
        cost_heatmap_origin = heatmap_origin + RIGHT * (n + 1) * heatmap_cell_size  # shift right

        #C = (X.reshape(-1, 1) - Y.reshape(1, -1)) ** 2  # In case not defined earlier

        for i in range(n):
            for j in range(n):
                val = C[i, j]
                color = interpolate_color(BLACK, WHITE, val / np.max(C))
                cell = Square(
                    side_length=heatmap_cell_size,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0
                )
                cell.move_to(
                    cost_heatmap_origin
                    + RIGHT * j * heatmap_cell_size
                    + DOWN * i * heatmap_cell_size
                )
                cost_heatmap.add(cell)

        C_label = MathTex(r"C = \text{Distance}^2").scale(0.7).next_to(cost_heatmap, UP)
        self.play(FadeIn(C_label), FadeIn(cost_heatmap), run_time=0.5)



        self.next_slide()
        self.next_slide(loop=True)
        print("looping through rows of T")
        # Animattion 44
        for i in range(n):
                # Rectangle highlight for the i-th row
                row_rect = Rectangle(
                    width=n * heatmap_cell_size,
                    height=heatmap_cell_size * 1.05,
                    stroke_color=YELLOW,
                    stroke_width=2
                )
                row_rect.move_to((heatmap_origin + RIGHT*1.8 + UP*0.06) + DOWN * (i + 0.5) * heatmap_cell_size)

                self.play(Create(row_rect), run_time=0.05)

                # Highlight source bar
                bar = source_hist.bars[i]
                highlight = bar.copy().set_stroke(YELLOW, width=2)
                self.play(Create(highlight), run_time=0.05)

                flows = VGroup()
                for j in range(n):
                    val = T_demo[i, j]
                    if val < 1e-5:
                        continue
                    height = val / np.max(b) * 3
                    flow = Rectangle(
                        width=0.1,
                        height=height,
                        fill_color=YELLOW,
                        fill_opacity=0.6,
                        stroke_width=0
                    )
                    target_bar = target_hist.bars[j]
                    flow.move_to(target_bar.get_bottom() + UP * height / 2)
                    self.add(flow)
                    flows.add(flow)

                self.wait(0.1)
                #self.next_slide()
                self.play(
                    FadeOut(highlight),
                    FadeOut(row_rect),
                    *[FadeOut(f) for f in flows],
                    run_time=0.05
                )

            

        #self.play(FadeOut(row_transport_title))
        self.next_slide()

        
        #self.clear()
        #####################################################################
        ####################################################################
        
                            # ENTROPY REGULARIZATION (Looping through gammas)
                            
         #####################################################################
        ####################################################################
        # Animation 42 
        self.clear()
        print("RENDERING ENTROPY REGULARIZATION")
        
        entropy_regularized = MathTex(
            r"T_\gamma^* = \underset{T}{\arg\min} \sum_{i,j} C_{ij} T_{ij} - \gamma H(T)"
        ).shift(UP*3, RIGHT*5).scale(0.6)
        
        matrix_constraint_a = MathTex(r"T \mathbf{1}_m = \vec{a}").scale(0.6).next_to(entropy_regularized, DOWN, buff=0.1)
        matrix_constraint_b = MathTex(r"T^T \mathbf{1}_n = \vec{b}").scale(0.6).next_to(matrix_constraint_a, DOWN, buff=0.3)
        
        
        self.play(FadeIn(entropy_regularized),
                  FadeIn(matrix_constraint_a),
                  FadeIn(matrix_constraint_b))
        
        
        # plane = NumberPlane(
        #     x_range=[-10, 10, 1],
        #     y_range=[-7, 7, 1],
        #     background_line_style={
        #         "stroke_color": BLUE,
        #         "stroke_width": 1,
        #         "stroke_opacity": 0.5,
        #     }
        # )
        # # Add the grid to the scene
        # self.add(plane)
        
        
        def sinkhorn(a, b, C, gamma, num_iters=100):
            """
            Compute Sinkhorn transport plan T for marginals a, b, cost matrix C,
            and entropy regularization parameter gamma.
            """
            K = np.exp(-C / gamma)
            u = np.ones_like(a)
            v = np.ones_like(b)
            
            for _ in range(num_iters):
                u = a / (K @ v)
                v = b / (K.T @ u)
                
            T = np.diag(u) @ K @ np.diag(v)
            return T




        n = 25
        x_vals = np.linspace(-4, 4, n)

        # Uniform source
        a = np.ones(n) / n

        # Gaussian target
        gaussian_pdf = np.exp(-0.5 * x_vals**2)
        b = gaussian_pdf / np.sum(gaussian_pdf)

        # Cost matrix: squared distances
        n = 25
        x_vals = np.linspace(-4, 4, n)
        a = np.ones(n) / n
        gaussian_pdf = np.exp(-0.5 * x_vals**2)
        b = gaussian_pdf / np.sum(gaussian_pdf)
        
        x_vals_src = np.linspace(-4, 0, n)
        x_vals_tgt = np.linspace(0, 4, n)
        X, Y = np.meshgrid(x_vals_src, x_vals_tgt)
        C = (X - Y)**2

        gammas = [0.00001, 0.01, .1, 1, 10, 100, 1000]

        square_size = 0.2
        origin = [UP * 2 + LEFT * 2.5]
        T_matrices = [sinkhorn(a, b, C, gamma) for gamma in gammas]
        max_val = np.max([T.max() for T in T_matrices])

        gaussian_bars = VGroup()
        uniform_bars = VGroup()

        bar_width = square_size * 0.9
        max_b = np.max(b)
        max_a = np.max(a)

        for idx, val in enumerate(b):
            bar_height = val / max_b  
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=RED,
                fill_opacity=0.8,
                stroke_width=0
            )
            bar.move_to(origin + UP * (bar_height/2 + 0.1) + RIGHT * idx * square_size)
            gaussian_bars.add(bar)

            uniform_bars = VGroup()

            for idx, val in enumerate(a):
                bar_length = val / max_a  
                bar = Rectangle(
                    width=bar_length,
                    height=bar_width,
                    fill_color=BLUE,
                    fill_opacity=0.8,
                    stroke_width=0
                )
               
                bar.move_to(
                    origin
                    + LEFT * (bar_length / 2 + 0.1)
                    + DOWN * idx * square_size
                )
                uniform_bars.add(bar)


        # Loop through gammas
        for gamma, T in zip(gammas, T_matrices):
            title = MathTex(rf"T^* \text{{at }} \gamma = {gamma}")\
                    .scale(0.8).to_edge(UP).shift(UP*0.2)

            squares = VGroup()
            local_max = T.max()

            for i in range(n):
                for j in range(n):
                    value = T[i, j]
                    scaled_value = value / local_max
                    color = interpolate_color(BLACK, WHITE, scaled_value)
                    square = Square(
                        side_length=square_size, fill_opacity=1,
                        fill_color=color, stroke_width=0
                    ).move_to(origin + RIGHT*j*square_size + DOWN*i*square_size)
                    squares.add(square)
                    
            # === Colorbar ===
            colorbar_height = n * square_size
            colorbar_width = 0.2
            n_steps = 50
            num_labels = 5 
            
            colorbar = VGroup()
            for k in range(n_steps):
                frac = k / (n_steps - 1)
                color = interpolate_color(BLACK, WHITE, frac)
                rect = Rectangle(
                    width=colorbar_width,
                    height=colorbar_height / n_steps,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0
                )
                rect.move_to(
                    origin 
                    + RIGHT * (n * square_size + 0.3)
                    + DOWN * (k + 0.5) * (colorbar_height / n_steps)
                )
                colorbar.add(rect)




            max_label = MathTex(f"{local_max:.4f}").scale(0.5).next_to(colorbar[-1], DOWN, buff=0.3)
            min_label = MathTex("0").scale(0.5).next_to(colorbar[0], UP, buff=0.3)

            


            self.play(
                FadeIn(squares, run_time=0.1),
                FadeIn(gaussian_bars, run_time=0.1),
                FadeIn(uniform_bars, run_time=0.1),
                FadeIn(title, run_time=0.1),
                FadeIn(max_label, run_time=0.1),
                FadeIn(min_label, run_time=0.1),
                FadeIn(colorbar, run_time=0.1)
            )

            self.next_slide()

            self.play(
                FadeOut(gaussian_bars, run_time=0.1), 
                FadeOut(uniform_bars, run_time=0.1),
                FadeOut(squares, run_time=0.1),
                FadeOut(title, run_time=0.1),
                FadeOut(max_label, run_time=0.1),
                FadeOut(min_label, run_time=0.1)
                #FadeOut(colorbar_group, run_time=0.1)
            )  
            
        self.next_slide()
        
        
        
        
            
        #####################################################################
        ####################################################################
        
                            # SINKHORN DERIVATION
                            
         #####################################################################
        ####################################################################
        
        
        
        #####################################################################
        ####################################################################
        
                            # SINKHORN ANIMATION
                            
         #####################################################################
        ####################################################################
        self.clear()
        #print("RENDERING APPLICATIONS IN ML")
        title = Tex("Sinkhorn Animation").scale(1.2)
        self.play(Write(title))


        #####################################################################
        ####################################################################
        
                            # APPLICATIONS IN ML
                            
         #####################################################################
        ####################################################################

        self.next_slide()
        self.clear()
        
        number_line = NumberLine(
            x_range=[-10, 10, 2],
            length=12,
            include_numbers=True,
            label_direction=DOWN
        )
        number_line.to_edge(UP)
        self.play(Write(number_line))
        

        P_rect = Rectangle(
            width=number_line.get_unit_size() * 4,  
            height=0.5,
            fill_color=RED,
            fill_opacity=0.5,
            stroke_color=RED
        )
        P_rect.move_to(number_line.number_to_point(-8) + UP * 0.5)
        P_label = Tex("P", color=RED).next_to(P_rect, UP)
        
        #  Q: uniform on [-2, 2]
        Q_rect = Rectangle(
            width=number_line.get_unit_size() * 4, 
            height=0.5,
            fill_color=GREEN,
            fill_opacity=0.5,
            stroke_color=GREEN
        )
        Q_rect.move_to(number_line.number_to_point(0) + UP * 0.5)
        Q_label = Tex("Q", color=GREEN).next_to(Q_rect, UP)
        
        # R: uniform on [6, 10]
        R_rect = Rectangle(
            width=number_line.get_unit_size() * 4,   
            height=0.5,
            fill_color=BLUE,
            fill_opacity=0.5,
            stroke_color=BLUE
        )
        R_rect.move_to(number_line.number_to_point(8) + UP * 0.5)
        R_label = Tex("R", color=BLUE).next_to(R_rect, UP)
        
        self.play(
            FadeIn(P_rect), FadeIn(P_label),
            FadeIn(Q_rect), FadeIn(Q_label),
            FadeIn(R_rect), FadeIn(R_label)
        )
        self.wait(1)
        

        num_bins = 100
        x = np.linspace(-10, 10, num_bins)
        

        P = np.where((x >= -10) & (x < -6), 1, 0).astype(np.float32)
        P = P / np.sum(P)
        
        # uniform on [-2, 2]
        Q = np.where((x >= -2) & (x < 2), 1, 0).astype(np.float32)
        Q = Q / np.sum(Q)
        
        # uniform on [6, 10]
        R = np.where((x >= 6) & (x <= 10), 1, 0).astype(np.float32)
        R = R / np.sum(R)
        

        X = x.reshape(-1, 1)
        Y = x.reshape(1, -1)
        C = (X - Y) ** 2  # shape: (num_bins, num_bins)
        

        def sinkhorn(a, b, C, gamma, num_iters=100):
            K = np.exp(-C / gamma)
            u = np.ones_like(a)
            v = np.ones_like(b)
            for _ in range(num_iters):
                u = a / (K @ v)
                v = b / (K.T @ u)
            T = np.outer(u, v) * K
            return T
        
        gamma = 1.0  
        T_PQ = sinkhorn(P, Q, C, gamma)
        T_PR = sinkhorn(P, R, C, gamma)
        T_QR = sinkhorn(Q, R, C, gamma)
        

        sink_PQ = np.sum(T_PQ * C)
        sink_PR = np.sum(T_PR * C)
        sink_QR = np.sum(T_QR * C)
        

        sinkhorn_formula = MathTex(
            r"D_{\text{Sink}}(P,Q)=\langle T^*, C\rangle"
        ).scale(0.7)
        cross_entropy_formula = MathTex(
            r"H(P,Q)=-\sum_i P_i \log Q_i"
        ).scale(0.7)
        kl_formula = MathTex(
            r"D_{\text{KL}}(P\parallel Q)=\sum_i P_i \log\frac{P_i}{Q_i}"
        ).scale(0.7)
        
        formulas = VGroup(sinkhorn_formula, cross_entropy_formula, kl_formula)
        formulas.arrange(RIGHT, aligned_edge=LEFT, buff=2.5).shift(UP * 2)
        #formulas.to_edge(LEFT, buff=1)
        formulas.shift(DOWN * 1.5)
        
        self.play(Write(formulas))
        self.wait(1)
       
        PQ_metrics = MathTex(
            rf"P\text{{ vs }}Q:\quad D_{{\text{{Sink}}}}={sink_PQ:.2f},\quad H=\infty,\quad D_{{\text{{KL}}}}=\infty"
        ).scale(0.7)

        PR_metrics = MathTex(
            rf"P\text{{ vs }}R:\quad D_{{\text{{Sink}}}}={sink_PR:.2f},\quad H=\infty,\quad D_{{\text{{KL}}}}=\infty"
        ).scale(0.7)

        QR_metrics = MathTex(
            rf"Q\text{{ vs }}R:\quad D_{{\text{{Sink}}}}={sink_QR:.2f},\quad H=\infty,\quad D_{{\text{{KL}}}}=\infty"
        ).scale(0.7)

        pairwise_metrics = VGroup(PQ_metrics, PR_metrics, QR_metrics)
        pairwise_metrics.arrange(DOWN, buff=0.5)
        pairwise_metrics.next_to(formulas, DOWN, buff=1)
        
        self.play(FadeIn(pairwise_metrics, shift=DOWN))
        self.wait(1)

        self.wait(3)