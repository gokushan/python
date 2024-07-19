import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

class FileUtilities:
    def __init__(self):
        self._supported_formats = {'png', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff', 'bmp'}
    
    def save_plot(self, figure: Figure, file_name: str, file_format: str, directory: str = '.', dpi: int = 300):
        """
        Guarda un gráfico en disco.

        :param figure: Una instancia de matplotlib.figure.Figure.
        :param file_name: Nombre del archivo sin la extensión.
        :param file_format: Formato del archivo (e.g., 'png', 'pdf', 'svg').
        :param directory: Directorio donde se guardará el archivo.
        :param dpi: Resolución del gráfico en puntos por pulgada (dpi).
        """
        if file_format.lower() not in self._supported_formats:
            raise ValueError(f"Formato de archivo no soportado: {file_format}. Los formatos soportados son: {self._supported_formats}")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, f"{file_name}.{file_format.lower()}")
        figure.savefig(file_path, format=file_format, dpi=dpi)