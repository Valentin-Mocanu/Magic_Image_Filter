# ----------------------------------------------
# GUI + LOGIC
# ----------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk # convert NumPy arrays to images and display them in Tkinter
import numpy as np
import time

from image_io import load_image, save_image
from gaussian_kernel import generate_gaussian_kernel
from convolution import convolve2d
from kernel_manager import save_kernel, load_kernel


class App:
    def __init__(self, root):

        # Main windows
        self.root = root
        self.root.title("Magic Image Filter")
        self.root.geometry("1400x700")

        # Original image
        self.image = None
        # Filtered image
        self.result = None
        # Image bit depth (8 or 24 bits)
        self.bit_depth = None
        # Kernel used for filtering
        self.kernel = None
        self.kernel_from_file = False # true = kernel loaded from file; false = kernel generated using kernel size + sigma (gaussian)


        # --------------------------
        # MAIN LAYOUT - 3 COLUMNS
        # --------------------------
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)


        # ---------------------------
        # FIRST COLUMN - CONTROL PANEL
        # ---------------------------
        control_frame = tk.LabelFrame(main_frame,
                                      text="Control Panel",
                                      padx=10,
                                      pady=10)
        control_frame.pack(side="left", fill="y", padx=15, pady=10)

        tk.Button(control_frame,
                  text="Load image",
                  width=20,
                  bg="#C2C2C2",
                  activebackground="#A8A8A8",
                  command=self.load_image_file).pack(pady=5)

        tk.Button(control_frame,
                  text="Save image",
                  width=20,
                  bg="#47ABFA",
                  activebackground="#4099E0",
                  command=self.save_image_file).pack(pady=5)

        # Spacing between buttons
        tk.Label(control_frame,text="").pack(pady=5)

        tk.Button(control_frame,
                  text="Load kernel",
                  width=20,
                  bg="#C2C2C2",
                  activebackground="#A8A8A8",
                  command=self.load_kernel_file).pack(pady=5)

        tk.Button(control_frame,
                  text="Save kernel",
                  width=20,
                  bg="#47ABFA",
                  activebackground="#4099E0",
                  command=self.save_kernel_file).pack(pady=5)

        tk.Label(control_frame, text="").pack(pady=5)

        # Kernel size
        tk.Label(control_frame, text="Kernel size (odd number):").pack()
        self.size_entry = tk.Entry(control_frame, width=10)
        self.size_entry.insert(0, "5")
        self.size_entry.pack(pady=5)

        # Sigma
        tk.Label(control_frame, text="Sigma:").pack()
        self.sigma_entry = tk.Entry(control_frame, width=10)
        self.sigma_entry.insert(0, "1.0")
        self.sigma_entry.pack(pady=5)

        tk.Label(control_frame, text="").pack(pady=5)

        # Padding
        tk.Label(control_frame, text="Padding type:").pack()

        self.padding_var = tk.StringVar(value="mirror")

        tk.Radiobutton(control_frame,
                       text="Zero Padding",
                       variable=self.padding_var,
                       value="zero").pack(anchor="w", padx=26)

        tk.Radiobutton(control_frame,
                       text="Mirror Padding",
                       variable=self.padding_var,
                       value="mirror").pack(anchor="w", padx=26)

        tk.Radiobutton(control_frame,
                       text="Replicate Padding",
                       variable=self.padding_var,
                       value="replicate").pack(anchor="w", padx=26)

        tk.Button(control_frame,
                  text="Apply filter",
                  width=20,
                  bg="#59CC5D",
                  activebackground="#4EB352",
                  command=self.apply_filter).pack(pady=10)

        tk.Button(control_frame,
                  text="Reset",
                  width=20,
                  bg="#f44336",
                  activebackground="#DB3C31",
                  command=self.reset_app).pack(pady=5)

        # Information panel
        self.info_frame = tk.Frame(control_frame)
        self.info_frame.pack(pady=10, fill="both", expand=True)

        self.info_scrollbar_y = tk.Scrollbar(self.info_frame, orient="vertical")
        self.info_scrollbar_y.pack(side="right", fill="y")

        self.info_scrollbar_x = tk.Scrollbar(self.info_frame, orient="horizontal")
        self.info_scrollbar_x.pack(side="bottom", fill="x")

        self.info = tk.Text(self.info_frame,
                            height=15,
                            width=32,
                            wrap="none",
                            yscrollcommand=self.info_scrollbar_y.set,
                            xscrollcommand=self.info_scrollbar_x.set)

        self.info.pack(side="left",
                       fill="both",
                       expand=True)

        self.info_scrollbar_y.config(command=self.info.yview)
        self.info_scrollbar_x.config(command=self.info.xview)


        # -------------------------------
        # SECOND COLUMN - ORIGINAL IMAGE
        # -------------------------------
        original_frame = tk.LabelFrame(main_frame,
                                       text="Original image",
                                       padx=10,
                                       pady=10)
        original_frame.pack(side="left", expand=True, fill="both", padx=15, pady=10)

        self.original_label = tk.Label(original_frame,
                                       text="Your image will appear here!",
                                       font=("Arial", 14),
                                       fg="gray")
        self.original_label.pack(expand=True)


        # ------------------------------
        # THIRD COLUMN - FILTERED IMAGE
        # ------------------------------
        result_frame = tk.LabelFrame(main_frame,
                                     text="Filtered image",
                                     padx=10,
                                     pady=10)
        result_frame.pack(side="left", expand=True, fill="both", padx=15, pady=10)

        self.result_label = tk.Label(result_frame,
                                     text="The filtered image will appear here!",
                                     font=("Arial", 14),
                                     fg="gray")
        self.result_label.pack(expand=True)


    # --------------------
    # LOAD IMAGE
    # --------------------
    def load_image_file(self):

        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.bmp *.jpeg")]
        )

        if not path:
            return

        self.image, self.bit_depth = load_image(path)
        self.result = None
        self.info.delete("1.0", tk.END)
        self.display_image(self.image, self.original_label)

        self.result_label.config(image="", text="The filtered image will appear here!", fg="gray")


    # -----------------
    # SAVE IMAGE
    # -----------------
    def save_image_file(self):

        if self.result is None:
            messagebox.showerror("Error", "You must first apply the filter to the uploaded image before you can save it!")
            return

        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            save_image(path, self.result, self.bit_depth)
            messagebox.showinfo("Save", "The image has been successfully saved!")


    # -------------------
    # DISPLAY IMAGE
    # -------------------
    def display_image(self, img_array, label):

        img = Image.fromarray(img_array.astype(np.uint8))
        img.thumbnail((500, 500))

        img_tk = ImageTk.PhotoImage(img)

        label.configure(image=img_tk, text="")
        label.image = img_tk


    # -------------------
    # LOAD KERNEL
    # -------------------
    def load_kernel_file(self):

        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")]
        )

        if not path:
            return

        try:
            self.kernel = load_kernel(path)
            self.kernel_from_file = True

            # Enable temporarily the text input (for Tkinter compatibility)
            self.size_entry.config(state="normal")
            self.sigma_entry.config(state="normal")

            size = self.kernel.shape[0]
            self.size_entry.delete(0, tk.END)
            self.size_entry.insert(0, str(size))

            self.sigma_entry.delete(0, tk.END)

            # Disable the text input if the user has loaded a kernel file
            self.size_entry.config(state="disabled")
            self.sigma_entry.config(state="disabled")

            messagebox.showinfo("Load", "The kernel has been successfully loaded!")

        except Exception as e:
            messagebox.showerror("Error", str(e))


    # -------------------
    # SAVE KERNEL
    # -------------------
    def save_kernel_file(self):

        if self.kernel is None:
            messagebox.showerror("Error", "No kernel has been generated or loaded!")
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt")

        if path:
            save_kernel(path, self.kernel)
            messagebox.showinfo("Save", "The kernel has been successfully saved!")


    # ------------------
    # APPLY FILTER
    # ------------------
    def apply_filter(self):

        if self.image is None:
            messagebox.showerror("Error", "You must load an image before you can start!")
            return

        try:
            padding = self.padding_var.get()

            # Generate a Gaussian kernel ONLY if no kernel has been loaded from a file
            if not self.kernel_from_file:
                size = int(self.size_entry.get())
                sigma = float(self.sigma_entry.get())

                if size % 2 == 0:
                    messagebox.showerror("Error", "The kernel size must be odd!")
                    return

                self.kernel = generate_gaussian_kernel(size, sigma)
            else:
                sigma = None

            kernel_used = self.kernel

            # Start measuring execution time
            start = time.time()

            # Convolution
            if len(self.image.shape) == 3:
                result = np.zeros_like(self.image, dtype=float)
                # -> for RGB
                for c in range(3):
                    result[:, :, c] = convolve2d(self.image[:, :, c], kernel_used, padding)
            else:
                # -> for grayscale
                result = convolve2d(self.image, kernel_used, padding)

            # Stop measuring execution time
            end = time.time()

            # Clamp the pixel values to the valid range (0–255)
            self.result = np.clip(result, 0, 255).astype(np.uint8)

            # Display the result
            self.display_image(self.result, self.result_label)
            self.info.delete("1.0", tk.END)  # Clear the previous text

            info_text = (
                f"Kernel size: {kernel_used.shape[0]}x{kernel_used.shape[1]}\n"
            )

            # Display the "sigma" only if a gaussian kernel has been generated
            if not self.kernel_from_file:
                info_text += f"Sigma: {sigma}\n"

            info_text += (
                f"Padding: {padding}\n"
                f"Execution time: {end - start:.4f} seconds\n"
                f"Kernel sum: {kernel_used.sum():.6f}\n"
                f"Kernel matrix values:\n"
                f"{np.array_str(kernel_used, precision=4, suppress_small=True)}"
            )

            self.info.insert(tk.END, info_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))


    # -------------------
    # RESET APP
    # -------------------
    def reset_app(self):

        self.image = None
        self.result = None
        self.kernel = None
        self.kernel_from_file = False

        # Enable the text input for kernel size and sigma
        self.size_entry.config(state="normal")
        self.sigma_entry.config(state="normal")

        # Reset image labels
        self.original_label.config(image="",
                                   text="Your image will appear here!",
                                   fg="gray")
        self.original_label.image = None

        self.result_label.config(image="",
                                 text="The filtered image will appear here!",
                                 fg="gray")
        self.result_label.image = None

        # Reset kernel values
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, "5")

        self.sigma_entry.delete(0, tk.END)
        self.sigma_entry.insert(0, "1.0")

        # Reset padding
        self.padding_var.set("mirror")

        # Clear information panel
        self.info.delete("1.0", tk.END)