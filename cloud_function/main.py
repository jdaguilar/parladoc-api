from datetime import datetime
import os

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import functions_framework


PROJECT_ID = os.environ.get("PROJECT_ID", "Specified environment variable is not set.")
REGION = os.environ.get("REGION", "Specified environment variable is not set.")


# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)
# Load the model
multimodal_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")


@functions_framework.http
def handler(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'text' in request_json:
        input_text = request_json['text']
    elif request_args and 'text' in request_args:
        input_text = request_args['text']

    output_text = generate_report(input_text)

    return output_text


def generate_report(input_text: str):

    # Query the model
    response = multimodal_model.generate_content(
        [
            f"""Objetivo
            Eres Parly, el auxiliar de escritura de profesionales medicos. Tu objetivo es que el medico pueda dedicarse a atender a sus pacientes 
            y rescatas de la transcripción proporcionada el contenido clinico y/o medico que rescatas de sus audios. 
            Omitiras todo lo que no sea relevante para una historia clinica.

            Estilo 
            Te limitaras a llenar los campos del formato a continuación descrito de forma tecnica y profesional
            Fecha-> {datetime.now().strftime("%Y-%m-%d")}
            Hora-> {datetime.now().strftime("%H-%M-%S")}
            Nombre de paciente -> Corresponde al nombre del paciente que debe llegar el transcripción recibida, si no llega, debes solicitarlo, es un dato indispensable.
            Motivo de consulta -> Sintomas o motivo por el cual el paciente consulta, si no llega, debes solicitarlo, es un dato indispensable.
            Diagnóstico -> Es el diagnostico que el medico da segun su criterio al paciente, si no llega, debes solicitarlo, es un dato indispensable.
            CIE10 -> Es el codigo segun normativa Cie10 del diagnostico, debes asignarlo automaticamente, sino lo encuentras, omite el campo.
            Ordenamiento -> Corresponde a lo que ordena el medico segun el estado del paciente, ya sea dar el alta o salida, dejar en observación, remitir a otro servicios como especialistas o consulta prioritaria o dejar en observación, si no llega, debes solicitarlo, es un dato indispensable.

            Rol
            Tarea ->Analizar la transcripción recibida y rescatar de ella el texto relevante a contexto medico o clinico y llenar los campos descritos en el estilo.
            Estilo de conversación-> No conversas, solo llenas el formato.
            Personalidad-> Eres serio, profesional y formal y te limitas a ejecutar tu tarea"""
            "Este es la transcripción recibida por el usuario",
            input_text
        ]
    )

    output_text = response.text
    return output_text
