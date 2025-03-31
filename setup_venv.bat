@echo off
REM Cria o ambiente virtual na pasta "venv"
python -m venv venv

REM Ativa o ambiente virtual
call venv\Scripts\activate

REM Instala as bibliotecas necessárias (TensorFlow e NumPy)
pip install tensorflow numpy

echo.
echo Configuração concluída! O ambiente virtual foi criado e os pacotes foram instalados.
echo Mantenha esta janela aberta para continuar trabalhando com o ambiente ativado.
pause
