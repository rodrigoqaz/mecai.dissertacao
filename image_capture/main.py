import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Imagem")
        
        self.code_label = tk.Label(root, text="Código do Fardinho:", font=('Arial', 25))
        self.code_label.pack()
        
        validate_cmd = self.root.register(self.validate_numeric_input)
        self.code_entry = tk.Entry(root, font="Arial 35 bold", validate="key", validatecommand=(validate_cmd, '%P'))
        self.code_entry.pack()
        
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()
        
        self.open_camera_button = tk.Button(self.buttons_frame, text="Abrir Câmera", command=self.open_camera, font=('Arial', 25))
        self.open_camera_button.grid(row=0, column=0, padx=10)
        
        self.rotate_button = tk.Button(self.buttons_frame, text="Girar Imagem", command=self.rotate_image, state=tk.DISABLED, font=('Arial', 25))
        self.rotate_button.grid(row=0, column=1, padx=10)
        
        self.capture_button = tk.Button(root, text="Capturar", command=self.capture_image, state=tk.DISABLED, font=('Arial', 25))
        self.capture_button.pack()
        
        self.camera_frame = tk.Label(root)
        self.camera_frame.pack()
        
        self.cap = None
        self.frame = None
        self.rotated_frame = None
        self.rotation_angle = 0
        
        self.root.bind('<space>', self.capture_image_event)

    def validate_numeric_input(self, input_value):
        """Função de validação para permitir apenas números."""
        if input_value.isdigit() or input_value == "":
            return True
        else:
            return False

    def open_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture_button.config(state=tk.NORMAL)
            self.rotate_button.config(state=tk.NORMAL)
            self.update_frame()
        else:
            messagebox.showinfo("Câmera já aberta", "A câmera já está aberta.")
    
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                if self.rotation_angle != 0:
                    (h, w) = self.frame.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
                    rotated = cv2.warpAffine(self.frame, M, (w, h))
                    cv2image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
                else:
                    cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_frame.imgtk = imgtk
                self.camera_frame.configure(image=imgtk)
            
            self.root.after(10, self.update_frame)

    def rotate_image(self):
        self.rotation_angle = (self.rotation_angle + 180) % 360 
    
    def capture_image(self):
        code = self.code_entry.get()
        if code:
            if self.rotation_angle != 0:
                (h, w) = self.frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
                rotated = cv2.warpAffine(self.frame, M, (w, h))
                filename = f"{code}.png"
                cv2.imwrite(filename, rotated)
            else:
                filename = f"{code}.png"
                cv2.imwrite(filename, self.frame)
            
            messagebox.showinfo("Sucesso", f"Imagem salva como {filename}")

            self.code_entry.delete(0, tk.END)
            self.root.focus_force()
            self.code_entry.focus()
            
        else:
            messagebox.showerror("Erro", "Por favor, insira um código.")
    
    def capture_image_event(self, event):
        """Função chamada quando a tecla espaço é pressionada."""
        self.capture_image()

    def close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
