
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

PATH_IMAGENS = 'image_capture/amostras/'

class AppCamera:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Imagem")
        
        # Insere código do fardinho
        self.etiqueta_codigo = tk.Label(root, text="Código do Fardinho:", font=('Arial', 25))
        self.etiqueta_codigo.pack()
        
        validar_comando = self.root.register(self.validar_entrada_numerica)
        self.entrada_codigo = tk.Entry(root, font="Arial 35 bold", validate="key", validatecommand=(validar_comando, '%P'))
        self.entrada_codigo.pack()
        
        # Controles da câmera
        self.quadro_botoes = tk.Frame(root)
        self.quadro_botoes.pack()
        
        self.botao_abrir_camera = tk.Button(self.quadro_botoes, text="Abrir Câmera", command=self.abrir_camera, font=('Arial', 25))
        self.botao_abrir_camera.grid(row=0, column=0, padx=10)
        
        self.botao_girar = tk.Button(self.quadro_botoes, text="Girar Imagem", command=self.girar_imagem, state=tk.DISABLED, font=('Arial', 25))
        self.botao_girar.grid(row=0, column=1, padx=10)
        
        # Controles de captura
        self.quadro_captura = tk.Frame(root)
        self.quadro_captura.pack()
        
        self.botao_capturar_am = tk.Button(self.quadro_captura, text="Amarelo", command=lambda: self.capturar_imagem_cor('AM'), font=('Arial', 25))
        self.botao_capturar_am.grid(row=0, column=0, padx=10)
        
        self.botao_capturar_br = tk.Button(self.quadro_captura, text="Branco", command=lambda: self.capturar_imagem_cor('BR'), font=('Arial', 25))
        self.botao_capturar_br.grid(row=0, column=1, padx=10)
        
        self.botao_capturar_ab = tk.Button(self.quadro_captura, text="Ambos", command=lambda: self.capturar_imagem_cor('AB', limpar=True), font=('Arial', 25))
        self.botao_capturar_ab.grid(row=0, column=2, padx=10)

        # Exibe imagem
        self.quadro_camera = tk.Label(root)
        self.quadro_camera.pack()
        
        self.cap = None
        self.quadro = None
        self.quadro_rotacionado = None
        self.angulo_rotacao = 0

        # Controles teclado
        self.root.bind('<space>', lambda event: self.capturar_imagem_cor('AB', limpar=True))
        self.root.bind('a', lambda event: self.capturar_imagem_cor('AM'))
        self.root.bind('b', lambda event: self.capturar_imagem_cor('BR'))

    def validar_entrada_numerica(self, valor_entrada):
        if valor_entrada.isdigit() or valor_entrada == "":
            return True
        else:
            return False

    def abrir_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.botao_capturar_am.config(state=tk.NORMAL)
            self.botao_capturar_br.config(state=tk.NORMAL)
            self.botao_capturar_ab.config(state=tk.NORMAL)
            self.botao_girar.config(state=tk.NORMAL)
            self.atualizar_quadro()

    def atualizar_quadro(self):
        if self.cap is not None and self.cap.isOpened():
            ret, self.quadro = self.cap.read()
            if ret:
                if self.angulo_rotacao != 0:
                    (h, w) = self.quadro.shape[:2]
                    centro = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(centro, self.angulo_rotacao, 1.0)
                    rotacionado = cv2.warpAffine(self.quadro, M, (w, h))
                    imagem_cv2 = cv2.cvtColor(rotacionado, cv2.COLOR_BGR2RGB)
                else:
                    imagem_cv2 = cv2.cvtColor(self.quadro, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(imagem_cv2)
                imgtk = ImageTk.PhotoImage(image=img)
                self.quadro_camera.imgtk = imgtk
                self.quadro_camera.configure(image=imgtk)
            
            self.root.after(10, self.atualizar_quadro)

    def girar_imagem(self):
        self.angulo_rotacao = (self.angulo_rotacao + 180) % 360 
    
    def capturar_imagem(self, cor, limpar=False):
        codigo = self.entrada_codigo.get()
        if codigo:
            if self.angulo_rotacao != 0:
                (h, w) = self.quadro.shape[:2]
                centro = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(centro, self.angulo_rotacao, 1.0)
                rotacionado = cv2.warpAffine(self.quadro, M, (w, h))
                nome_arquivo = f"{PATH_IMAGENS}{cor}_{codigo}.png"
                cv2.imwrite(nome_arquivo, rotacionado)
            else:
                nome_arquivo = f"{PATH_IMAGENS}{cor}_{codigo}.png"
                cv2.imwrite(nome_arquivo, self.quadro)
            
            messagebox.showinfo("Sucesso", f"Imagem salva como {nome_arquivo}")
            self.root.focus_force()

            if limpar:
                self.entrada_codigo.delete(0, tk.END)
                self.root.focus_force()
                self.entrada_codigo.focus()
            
        else:
            messagebox.showerror("Erro", "Por favor, insira um código.")
    
    def capturar_imagem_cor(self, cor, limpar=False):
        self.capturar_imagem(cor, limpar)

    def fechar(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = AppCamera(root)
    root.protocol("WM_DELETE_WINDOW", app.fechar)
    root.mainloop()
