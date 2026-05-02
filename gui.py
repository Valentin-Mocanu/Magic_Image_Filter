# ----------------------------------------------
# INTERFATA GRAFICA (GUI) + LOGICA PRINCIPALA
# ----------------------------------------------

import tkinter as tk # biblioteca pentru GUI
from tkinter import filedialog, messagebox # deschidere fisiere si messagebox eroare/informare
from PIL import Image, ImageTk # conversie numpy in imagine si afisare in Tkinter
import numpy as np # lucrul cu matrice
import time # masurare timp executie

from image_io import load_image, save_image
from gaussian_kernel import generate_gaussian_kernel
from convolution import convolve2d
from kernel_manager import save_kernel, load_kernel


class App:

    # Constructor
    def __init__(self, root):

        # Cream fereastra
        self.root = root
        self.root.title("Filtrarea Gaussiana a Imaginilor")
        self.root.geometry("1400x700")

        # imaginea originala
        self.image = None
        # imaginea blurata
        self.result = None
        # bit_depth 8 sau 24
        self.bit_depth = None
        # nucleul Gaussian
        self.kernel = None

        # --------------------------
        # MAIN LAYOUT - 3 COLOANE
        # --------------------------
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # ---------------------------
        # COLOANA 1 - CONTROL PANEL
        # ---------------------------
        control_frame = tk.LabelFrame(main_frame,
                                      text="Control Panel",
                                      padx=10,
                                      pady=10)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Butoane pentru incarcarea unei imagini si salvarea imaginii rezultate
        tk.Button(control_frame, text="Incarca Imagine",
                  width=20,
                  command=self.load).pack(pady=5)

        tk.Button(control_frame, text="Salveaza Imagine",
                  width=20,
                  command=self.save_image).pack(pady=5)

        # Spatiere intre butoane
        tk.Label(control_frame, text="").pack(pady=5)

        # Butoane pentru incarcarea/salvarea kernel-ului (nucleului)
        tk.Button(control_frame, text="Incarca Kernel",
                  width=20,
                  command=self.load_kernel_file).pack(pady=5)

        tk.Button(control_frame, text="Salveaza Kernel",
                  width=20,
                  command=self.save_kernel_file).pack(pady=5)

        tk.Label(control_frame, text="").pack(pady=5)

        # Dimensiune Kernel
        tk.Label(control_frame, text="Dimensiune Kernel (impar):").pack()
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
        tk.Label(control_frame, text="Tip Padding:").pack()

        self.padding_var = tk.StringVar(value="mirror")

        tk.Radiobutton(control_frame, text="Zero Padding",
                       variable=self.padding_var,
                       value="zero").pack(anchor="w")

        tk.Radiobutton(control_frame, text="Mirror Padding",
                       variable=self.padding_var,
                       value="mirror").pack(anchor="w")

        tk.Radiobutton(control_frame, text="Replicate Padding",
                       variable=self.padding_var,
                       value="replicate").pack(anchor="w")

        # Buton de rulare
        tk.Button(control_frame,
                  text="Aplica Filtru",
                  width=20,
                  bg="#4CAF50",
                  fg="white",
                  command=self.apply_filter).pack(pady=10)

        # Buton resetare
        tk.Button(control_frame,
                  text="Reset",
                  width=20,
                  bg="#f44336",
                  fg="white",
                  command=self.reset_app).pack(pady=5)

        # Info cu bara de derulare
        self.info_frame = tk.Frame(control_frame)
        self.info_frame.pack(pady=10, fill="both", expand=True)

        self.info_scrollbar = tk.Scrollbar(self.info_frame)
        self.info_scrollbar.pack(side="right", fill="y")

        self.info = tk.Text(self.info_frame,
                            height=15, # Inaltimea vizibila
                            width=30, # Latimea widget-ului
                            wrap="none", # Nu face wrap automat
                            yscrollcommand=self.info_scrollbar.set)
        self.info.pack(side="left", fill="both", expand=True)

        self.info_scrollbar.config(command=self.info.yview)

        # -------------------------------
        # COLOANA 2 - IMAGINEA ORIGINALA
        # -------------------------------
        original_frame = tk.LabelFrame(main_frame,
                                       text="Imagine Originala",
                                       padx=10,
                                       pady=10)
        original_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

        self.original_label = tk.Label(original_frame,
                                       text="Incarca o imagine pentru a incepe",
                                       font=("Arial", 14),
                                       fg="gray")
        self.original_label.pack(expand=True)

        # ------------------------------
        # COLOANA 3 - IMAGINEA BLURATA
        # ------------------------------
        result_frame = tk.LabelFrame(main_frame,
                                     text="Imagine Blurata",
                                     padx=10,
                                     pady=10)
        result_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

        self.result_label = tk.Label(result_frame,
                                     text="Imaginea blurata va aparea aici",
                                     font=("Arial", 14),
                                     fg="gray")
        self.result_label.pack(expand=True)

    # --------------------
    # INCARCARE IMAGINE
    # --------------------
    def load(self):

        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.bmp *.jpeg")]
        )

        if not path:
            return

        self.image, self.bit_depth = load_image(path)
        self.display_image(self.image, self.original_label)

        self.result_label.config(image="", text="Imaginea blurata va aparea aici", fg="gray")

    # -------------------
    # AFISARE IMAGINE
    # -------------------
    def display_image(self, img_array, label):

        img = Image.fromarray(img_array.astype(np.uint8))
        img.thumbnail((500, 500))

        img_tk = ImageTk.PhotoImage(img)

        label.configure(image=img_tk, text="")
        label.image = img_tk

    # -------------------
    # RESET APLICATIE
    # -------------------
    def reset_app(self):

        # Reset imagini
        self.image = None
        self.result = None
        self.kernel = None

        # Reset labels imagine
        self.original_label.config(image="",
                                   text="Incarca o imagine pentru a incepe",
                                   fg="gray")
        self.original_label.image = None

        self.result_label.config(image="",
                                 text="Imaginea blurata va aparea aici",
                                 fg="gray")
        self.result_label.image = None

        # Reset valori kernel
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, "5")

        self.sigma_entry.delete(0, tk.END)
        self.sigma_entry.insert(0, "1.0")

        # Reset padding
        self.padding_var.set("mirror")

        # Golire info
        self.info.delete("1.0", tk.END)

    # ------------------
    # APLICARE FILTRU
    # ------------------
    def apply_filter(self):

        # Verificam daca exista imaginea
        if self.image is None:
            messagebox.showerror("Eroare", "Incarca o imagine mai intai!")
            return

        try:
            # Citim parametrii din UI
            size = int(self.size_entry.get())
            sigma = float(self.sigma_entry.get())

            # Verificam daca kernel-ul (nucleul) este impar
            if size % 2 == 0:
                messagebox.showerror("Eroare", "Dimensiunea kernelului trebuie sa fie impara!")
                return

            padding = self.padding_var.get()

            # Kernel-ul (nucleul) folosit la rulare
            if self.kernel is None:
                # Daca nu exista kernel incarcat din fisier, generam unul pe baza UI
                kernel_used = generate_gaussian_kernel(size, sigma)
            else:
                # Daca kernelul a fost incarcat, il folosim pe acesta
                kernel_used = self.kernel

            # Masuram timpul
            start = time.time()

            # Convolutie
            if len(self.image.shape) == 3:
                result = np.zeros_like(self.image, dtype=float)
                # -> pentru RGB
                for c in range(3):
                    result[:, :, c] = convolve2d(self.image[:, :, c], kernel_used, padding)
            else:
                # -> pentru grayscale
                result = convolve2d(self.image, kernel_used, padding)

            # Oprim masurarea timpului
            end = time.time()

            # Limitam valorile (evitam depasirea intervalului 0–255)
            self.result = np.clip(result, 0, 255).astype(np.uint8)

            # Afisam rezultatul
            self.display_image(self.result, self.result_label)

            # ------------------------------
            # Afisam informatiile cu scroll
            # ------------------------------
            self.info.delete("1.0", tk.END)  # șterge textul anterior
            self.info.insert(tk.END, f"""Dimensiune kernel: {kernel_used.shape[0]}x{kernel_used.shape[1]}
Sigma: {sigma}
Padding: {padding}
Timp executie: {end - start:.4f} secunde
Suma kernel: {kernel_used.sum():.6f}
Valori matrice kernel:
{np.array_str(kernel_used, precision=4, suppress_small=True)}
""")

        except Exception as e:
            messagebox.showerror("Eroare", str(e))


    # -----------------
    # SALVARE IMAGINE
    # -----------------
    def save_image(self):

        if self.result is None:
            messagebox.showerror("Eroare", "Nu exista rezultat!")
            return

        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            save_image(path, self.result, self.bit_depth)
            messagebox.showinfo("Succes", "Imagine salvata!")

    # -------------------
    # SALVARE KERNEL
    # -------------------
    def save_kernel_file(self):

        if self.kernel is None:
            messagebox.showerror("Eroare", "Nu exista kernel generat sau incarcat!")
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt")

        if path:
            save_kernel(path, self.kernel)
            messagebox.showinfo("Succes", "Kernel salvat!")

    # -------------------
    # INCARCARE KERNEL
    # -------------------
    def load_kernel_file(self):

        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")]
        )

        if not path:
            return

        try:
            self.kernel = load_kernel(path)

            messagebox.showinfo("Succes", "Kernel incarcat!")

            size = self.kernel.shape[0]
            self.size_entry.delete(0, tk.END)
            self.size_entry.insert(0, str(size))

        except Exception as e:
            messagebox.showerror("Eroare", str(e))