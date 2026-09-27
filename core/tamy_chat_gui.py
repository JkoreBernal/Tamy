# tamy_chat_gui.py - Interfaz Gráfica para TAMY
import customtkinter as ctk
from tkinter import filedialog
import os

import tamy_brain

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class TamyGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚡ TAMY // TACTICAL ASSISTANT")
        self.geometry("600x650")
        self.resizable(True, True)

        self._crear_interfaz()

    def _crear_interfaz(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#161b22", corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        title = ctk.CTkLabel(header, text="🤖 TAMY CORE v1.0", font=ctk.CTkFont(size=18, weight="bold"), text_color="#58a6ff")
        title.pack(side="left", padx=20, pady=15)

        sub = ctk.CTkLabel(header, text="🟢 ONLINE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3fb950")
        sub.pack(side="right", padx=20, pady=15)

        # Consola / Chat Output
        self.chat_display = ctk.CTkTextbox(self, fg_color="#0d1117", text_color="#c9d1d9", font=ctk.CTkFont(family="Consolas", size=12))
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        self._imprimir_sistema("TAMY inicializada. Escribe un mensaje o sube una imagen/captura para analizar.")

        # Panel de Controles
        control_frame = ctk.CTkFrame(self, fg_color="#161b22")
        control_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.entry_msg = ctk.CTkEntry(control_frame, placeholder_text="Escribe un mensaje o comando...", width=380)
        self.entry_msg.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        self.entry_msg.bind("<Return>", lambda event: self._enviar_mensaje())

        btn_send = ctk.CTkButton(control_frame, text="Enviar", width=80, command=self._enviar_mensaje, fg_color="#238636", hover_color="#2ea043")
        btn_send.pack(side="left", padx=(0, 5), pady=10)

        btn_img = ctk.CTkButton(control_frame, text="📷 Imagen", width=90, command=self._subir_imagen, fg_color="#1f6beb", hover_color="#388bfd")
        btn_img.pack(side="left", padx=(0, 10), pady=10)

    def _imprimir_sistema(self, texto):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{texto}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _imprimir_usuario(self, texto):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"👤 Tú: {texto}\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _enviar_mensaje(self):
        msg = self.entry_msg.get().strip()
        if not msg:
            return

        self._imprimir_usuario(msg)
        self.entry_msg.delete(0, "end")

        respuesta = tamy_brain.procesar_texto(msg)
        self._imprimir_sistema(f"🤖 TAMY: {respuesta}")

    def _subir_imagen(self):
        ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar Imagen para TAMY",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.webp")]
        )
        if ruta_archivo:
            comentario = self.entry_msg.get().strip()
            self._imprimir_usuario(f"[Imagen cargada: {os.path.basename(ruta_archivo)}]")
            if comentario:
                self.entry_msg.delete(0, "end")

            respuesta = tamy_brain.analizar_imagen_y_aprender(ruta_archivo, comentario)
            self._imprimir_sistema(f"🤖 TAMY: {respuesta}")


if __name__ == "__main__":
    app = TamyGUI()
    app.mainloop()