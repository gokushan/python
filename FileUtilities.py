import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import requests
from pathlib import Path

class SaveFile:

    """
    Clase para guardar un archivo gráfico en disco en el directorio especificado y de alguno de los formatos indicados en supported_formats

    Métodos:
    save_plot : Guarda el archivo en disco.
    """
      
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

        return True

class DownloadFiles:
    """
    Clase para descargar archivos desde URLs y guardarlos en un directorio especificado.

    Métodos:
    download_file: Descarga un archivo a partir de la url y el nombre.
    download_files: Descarga una lista de archivos indicados en una lista de urls
    """

    def __init__(self, download_path):
        """
        Inicializa la clase DownloadFiles con el directorio de destino para los archivos descargados.

        Parameters:
        download_path (str): Directorio donde se guardarán los archivos descargados.
        """
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename=None):
        """
        Descarga un archivo de una URL específica y lo guarda en el directorio de destino.

        Parameters:
        url (str): URL del archivo a descargar.
        filename (str, optional): Nombre del archivo. Si no se proporciona, se usa el nombre original del archivo.

        Returns:
        Path: Ruta completa del archivo descargado.
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Levanta una excepción si la descarga falla
        
        if filename is None:
            filename = url.split('/')[-1]  # Obtiene el nombre del archivo de la URL
        
        file_path = self.download_path / filename
        
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        return file_path

    def download_files(self, urls):
        """
        Descarga un conjunto de archivos de una lista de URLs.

        Parameters:
        urls (list of str): Lista de URLs de los archivos a descargar.

        Returns:
        list of Path: Lista de rutas completas de los archivos descargados.
        """
        downloaded_files = []
        for url in urls:
            try:
                file_path = self.download_file(url)
                downloaded_files.append(file_path)
                print(f"Descargado: {file_path}")
            except Exception as e:
                print(f"Error al descargar {url}: {e}")
        
        return downloaded_files
