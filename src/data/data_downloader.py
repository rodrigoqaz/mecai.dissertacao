import os
import io
import zipfile
import logging
import shutil
import pickle
import yaml
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveZipDownloader:
    def __init__(self, credentials_path='credentials.json', token_path='token.pickle'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None

    def authenticate_drive(self):
        creds = None
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        self.service = build('drive', 'v3', credentials=creds)

    def get_zip_file_id(self, folder_id, zip_name=None):
        query = f"'{folder_id}' in parents and trashed = false"
        if zip_name:
            query += f" and name = '{zip_name}'"
        else:
            query += " and mimeType = 'application/zip'"
        try:
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            if not files:
                logging.warning("Nenhum arquivo ZIP encontrado.")
                return None
            zip_file = files[0]
            logging.info(f"Arquivo ZIP encontrado: {zip_file['name']}")
            return zip_file['id'], zip_file['name']
        except Exception as e:
            logging.error(f"Erro ao buscar arquivo ZIP: {e}")
            return None

    def download_and_extract(self, file_id, zip_name, output_dir):
        logging.info(f"Iniciando download do arquivo: {zip_name}")
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        zip_path = os.path.join(temp_dir, zip_name)
        try:
            file_metadata = self.service.files().get(fileId=file_id, fields="size").execute()
            total_size = int(file_metadata.get("size", 0))
            request = self.service.files().get_media(fileId=file_id)
            with io.FileIO(zip_path, 'wb') as fh, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=f"Baixando {zip_name}"
                ) as pbar:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        pbar.update(status.resumable_progress - pbar.n)
            logging.info(f"Download concluído: {zip_path}")
            logging.info(f"Extraindo conteúdo para: {output_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.infolist():
                    relative_path = os.path.relpath(
                        member.filename,
                        (
                            member.filename.split(os.sep, 1)[0]
                            if os.sep in member.filename
                            else ""
                        ),
                    )
                    if member.is_dir():
                        os.makedirs(
                            os.path.join(output_dir, relative_path), exist_ok=True
                        )
                    elif relative_path:
                        with zip_ref.open(member) as source, open(
                            os.path.join(output_dir, relative_path), "wb"
                        ) as target:
                            target.write(source.read())
                # zip_ref.extractall(output_dir)
            logging.info("Extração concluída!")
        except Exception as e:
            logging.error(f"Erro durante download ou extração: {e}")
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def download_zip_from_drive(self, folder_id, output_dir, zip_name=None):
        self.authenticate_drive()
        result = self.get_zip_file_id(folder_id, zip_name)
        if result is None:
            logging.error("Não foi possível encontrar o arquivo ZIP para download.")
            return
        file_id, file_name = result
        os.makedirs(output_dir, exist_ok=True)
        self.download_and_extract(file_id, file_name, output_dir)

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config_path = "src/config/data/download.yaml"
    config = load_config(config_path)

    downloader = GoogleDriveZipDownloader(
        credentials_path=config.get("credentials", "credentials.json"),
        token_path=config.get("token", "token.pickle")
    )
    downloader.download_zip_from_drive(
        folder_id=config["folder_id"],
        output_dir=config.get("output_dir", "data/bronze"),
        zip_name=config.get("zip_name")
    )
