# ProyectoGPU

### Proyecto de Simulación de partículas. 

Para ejecutar el programa abrir archivo `CudaProyect.exe`.

### Configuraciones:

Las variables de gravedad, coeficiente de roce y número de pertículas han sido definidas como:
  - Gravedad: -9.8
  - Coeficiente de roce del aire: 0.32
  - Número de pertículas: 500
  
 En caso de querer cambiar las configuraciones se deben cambiar los valores en las lineas 72,73 y 75 respectivamente en el archivo `kernel.cu`.
 
 ### Herramientas:
 
 Este proyecto debe ejecutarse bajo Visual Studio 2022 y usando Cuda Toolkit 11.7.
 
 ### Librerias:
 
 Las librerias utilizadas para la configuración del proyecto son:
  - OpenGL
  - GLM 0.9.9.7
  - GLEW 2.1.0
  - GLFW3 3.3.7
