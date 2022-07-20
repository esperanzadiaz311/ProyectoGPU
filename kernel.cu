// Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cmath>
//#include <shader.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <vector>
#include "shape.h"
#include "transform3d.h"
#include "shader2.h"
#include<time.h>

using namespace glm;
using namespace std;

/*
    Aquí va el kernel :D
*/

__global__ void kernel_particula(int particles_num, vec3* particlesPositions, vec3* particlesNewPositions, 
    vec3* particlesVelocities, vec3* particlesNewVelocities, float radio) {
    int particleIndex = threadIdx.x + blockIdx.x * blockDim.x;
    bool collision = false;
    for (int i = 0; i < particles_num; i++) {
        vec3 posActual = particlesPositions[particleIndex];
        vec3 posI = particlesPositions[i];
        vec3 velActual = particlesVelocities[particleIndex];
        vec3 velI = particlesVelocities[i];
        if(i != particleIndex){
           
            double distance = sqrt(pow(posActual[0] - posI[0], 2.0) + pow(posActual[1] -  posI[1], 2.0) + pow(posActual[2] - posI[2], 2.0));
            if (distance <= radio * 2.0f) {
                collision = true;
                vec3 normal = posI - posActual;
                /*
                normal[0] = posI[0] - posActual[0];
                normal[1] = posI[1] - posActual[1];
                normal[2] = posI[2] - posActual[2];
                */
                normal = normalize(normal);
                /*
                float modulo = sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2));
                normal[0] = normal[0] / modulo;
                normal[1] = normal[1] / modulo;
                normal[2] = normal[2] / modulo;
                */

                vec3 relativeVel = velI - velActual;
                /*
                relativeVel[0] = velI[0] - velActual[0];
                relativeVel[1] = velI[1] - velActual[1];
                relativeVel[2] = velI[2] - velActual[2];
                */
                float prod_punto = dot(normal, relativeVel);//normal[0] * relativeVel[0] + normal[1] * relativeVel[1] + normal[2] * relativeVel[2];
                normal *= prod_punto;
                vec3 normalVelocity = normal;//vec3(normal[0] * prod_punto, normal[1] * prod_punto, normal[2] * prod_punto);
                vec3 velocityRes = velActual + normalVelocity;//vec3(velActual[0] + normalVelocity[0], velActual[1] + normalVelocity[1], velActual[2] + normalVelocity[2]);
                particlesNewVelocities[particleIndex] = velocityRes;

                break;
            }
        }
        if(!collision){
            particlesNewVelocities[particleIndex] = velActual;
            particlesNewPositions[particleIndex] = posActual;
        }

    }
}

int particles_num = 100;
const float gravity = -20.8f;
const float radio = 0.1f;
const float coeficienteRoce = 0.8f;
    
/*
    Load an OBJ file. Only triangular faces.
*/

bool loadOBJ(
    const char* path,
    vector < float >& out_vertices,
    vector < vec2 >& out_uvs,
    vector < vec3 >& out_normals,
    vector < unsigned int >& out_indexs
){
    vector< unsigned int > vertexIndices, uvIndices, normalIndices;
    vector< vec3 > temp_vertices;
    vector< vec2 > temp_uvs;
    vector< vec3 > temp_normals;

    
    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return false;
    }
    unsigned int index = 0;
    while (true) {

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF) break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader
        if (strcmp(lineHeader, R"(v)") == 0) {
            vec3 vertex;
            fscanf(file, R"(%f %f %f)", &vertex.x, &vertex.y, &vertex.z);
            temp_vertices.push_back(vertex);
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            vec2 uv;
            fscanf(file, R"(%f %f\n)", &uv.x, &uv.y);
            temp_uvs.push_back(uv);
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            vec3 normal;
            fscanf(file, R"(%f %f %f\n)", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(normal);
        }
        else if (strcmp(lineHeader, "f") == 0) {
            //string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
            if (matches != 9) {
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            out_indexs.push_back(index);
            out_indexs.push_back(index+1);
            out_indexs.push_back(index+2);
            uvIndices.push_back(uvIndex[0]);
            uvIndices.push_back(uvIndex[1]);
            uvIndices.push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
            index += 3;
        }
    }
    // For each vertex of each triangle
    for (unsigned int i = 0; i < vertexIndices.size(); i++) {
        unsigned int vertexIndex = vertexIndices[i];
        vec3 vertex = temp_vertices[vertexIndex - 1];
        out_vertices.push_back(vertex[0]);
        out_vertices.push_back(vertex[1]);
        out_vertices.push_back(vertex[2]);
        out_vertices.push_back(0.9);
        out_vertices.push_back(0.3);
        out_vertices.push_back(0.5);
        
        
    }
}


vec3 randomVec3(float vmin,float vmax) {
    float randX = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    float randY = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    float randZ = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    //cout << ' ' << randX << ',' << randY << ',' << randZ << endl;
    return vec3(randX, randY, randZ);
}

vec3 calculateNewPos(vec3 pos0,vec3 vel0, float dt, float a) {
    vec3 newVel = vec3(0.0f, 0.0f, 0.0f);
    newVel[0] = vel0[0] * (1 - coeficienteRoce *dt);
    newVel[1] = (vel0[1] + (a * dt)) * (1 - coeficienteRoce * dt);
    newVel[2] = vel0[2] * (1 - coeficienteRoce * dt);
     //sqrt(pow(vel0, vec3(2, 2, 2)) + aceleracion);
    /*
    float newX = pos0[0] + vel0[0] * dt;
    float newY = pos0[1] + vel0[1] * dt + 0.5*a*pow(dt,2.0);
    float newZ = pos0[2] + vel0[2] * dt;
    */
    newVel *= dt;
    vec3 newPos = pos0 + newVel;
    return newPos;
}

vec3 calculateTranslate(vec3 vel0, float dt) {
    float newX = vel0[0] * dt;
    float newY = vel0[1] * dt;
    float newZ = vel0[2] * dt;
    vec3 newPos = vec3(newX,newY,newZ);
    return newPos;
}

GLFWwindow* window;

Shape* square;
Shape* square2;
Shape* sphere;

// The transform being used to draw our shape
Transform3D transformCube;
Transform3D transformParticle1;

// These shader objects wrap the functionality of loading and compiling shaders from files.
Shader vertexShader;
Shader fragmentShader;

// GL index for shader program
GLuint shaderProgram;

// Index of the world matrix in the vertex shader.
GLuint worldMatrixUniform;

// Index of the camera matrix in the vertex shader.
GLuint cameraMatrixUniform;

// Here we store the position, of the camera.
Transform3D cameraPosition;

// Store the current dimensions of the viewport.
vec2 viewportDimensions = vec2(800, 600);

// Store the current mouse position.
vec2 mousePosition;

// Window resize callback
void resizeCallback(GLFWwindow* window, int width, int height){
	glViewport(0, 0, width, height);
    viewportDimensions = vec2(width, height);
}

// This will get called when the mouse moves.
void mouseMoveCallback(GLFWwindow *window, GLdouble mouseX, GLdouble mouseY)
{
    mousePosition = vec2(mouseX, mouseY);
}

/*
    Particles Elements
*/
vector<unsigned int> count(particles_num,0);
vector<vector<unsigned int>> particlesID;
vector<vec3> particlesPosition;
vector<vec3> particlesVelocity;
vector<vec3> particlesColor;
vector<Shape*> particlesShapes;
vector<Transform3D> particlesTransforms;

void particleInit(){
    srand(time(NULL));
    vector< float > verticesOBJ;
    vector< vec3 > colors;
    vector< vec2 > uvs;
    vector< vec3 > normals; // Won't be used at the moment.
    vector< unsigned int > indicesOBJ;
    bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    for (int i=0;i<particles_num;i++){
        particlesPosition.push_back(randomVec3(-1.0f,1.0f));
        particlesVelocity.push_back(randomVec3(-0.5f,0.5f));
        Transform3D transform;
        particlesTransforms.push_back(transform);
        particlesTransforms[i].SetPosition(particlesPosition[i]);
        cout << i << ' ' << particlesTransforms[i].Position()[0] << ',' << particlesTransforms[i].Position()[1] << ',' << particlesTransforms[i].Position()[2] << endl;
        //particlesTransforms[i].Translate(vec3(0, 0, -5));
        Shape* sphere = new Shape(verticesOBJ, indicesOBJ);
        particlesShapes.push_back(sphere);
        particlesColor.push_back(randomVec3(0.0f,1.0f));        
    }
}
/*
void genParticle() {
    vector< vec3 > verticesOBJ;
    vector< vec3 > colors;
    vector< vec2 > uvs;
    vector< vec3 > normals; // Won't be used at the moment.
    vector< unsigned int > indicesOBJ;
    bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    particlesPosition.push_back(vec3(0.5f,0.8f,0.5f));
    particlesVelocity.push_back(vec3(0,0,0));
    Transform3D transform;
    particlesTransforms.push_back(transform);
    particlesTransforms[size(particlesTransforms)-1].SetPosition(particlesPosition[size(particlesPosition) - 1]);
    cout << size(particlesTransforms) - 1 << ' ' << particlesTransforms[size(particlesTransforms) - 1].Position()[0] << ',' << particlesTransforms[size(particlesTransforms) - 1].Position()[1] << ',' << particlesTransforms[size(particlesTransforms) - 1].Position()[2] << endl;
    //particlesTransforms[i].Translate(vec3(0, 0, -5));
    Shape* sphere = new Shape(verticesOBJ, indicesOBJ);
    particlesShapes.push_back(sphere);
    particlesColor.push_back(randomVec3(0.0f, 1.0f));
    particles_num++;
}
*/

int main(int argc, char **argv)
{
	// Initializes the GLFW library
	glfwInit();

	// Initialize window
	GLFWwindow* window = glfwCreateWindow(viewportDimensions.x, viewportDimensions.y, "Proyecto GPU", nullptr, nullptr);

	glfwMakeContextCurrent(window);

	//set resize callback
	glfwSetFramebufferSizeCallback(window, resizeCallback);

    glfwSetCursorPosCallback(window, mouseMoveCallback);

	// Initializes the glew library
	glewInit();


	// Indices for cube (-1, -1, -1) to (1, 1, 1)
    //    [2]------[6]
	// [3]------[7] |
	//	|  |     |  |
	//	|  |     |  |
	//	| [0]----|-[4]
	// [1]------[5]

	// Create square vertex data.
	vector<float> vertices;
    
    vertices.push_back(-1.0f); 
    vertices.push_back(-1.0f); 
    vertices.push_back(-1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
   

    vertices.push_back(-1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);

	//vertices.push_back(vec3(-1.0f, 1.0f, -1.0f));
    vertices.push_back(-1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
	//vertices.push_back(vec3(-1.0f, 1.0f, 1.0f));
    vertices.push_back(-1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
    //vertices.push_back(vec3(1.0f, -1.0f, -1.0f));
    vertices.push_back(1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
    //vertices.push_back(vec3(1.0f, -1.0f, 1.0f));
    vertices.push_back(1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
    //vertices.push_back(vec3(1.0f, 1.0f, -1.0f));
    vertices.push_back(1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
    //vertices.push_back(vec3(1.0f, 1.0f, 1.0f));
    vertices.push_back(1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.9f);
    vertices.push_back(0.3f);
    vertices.push_back(0.8f);
    
    /*
    vector<vec3> vertices2;
    vertices2.push_back(vec3(-0.5f, -0.5f, -0.5f));
	vertices2.push_back(vec3(-0.5f, -0.5f, 0.5f));
	vertices2.push_back(vec3(-0.5f, 0.5f, -0.5f));
	vertices2.push_back(vec3(-0.5f, 0.5f, 0.5f));
    vertices2.push_back(vec3(0.5f, -0.5f, -0.5f));
    vertices2.push_back(vec3(0.5f, -0.5f, 0.5f));
    vertices2.push_back(vec3(0.5f, 0.5f, -0.5f));
    vertices2.push_back(vec3(0.5f, 0.5f, 0.5f));
    */
   
    
	vector<unsigned int> indices;
    for (int i = 0; i < 2; i++)
    {
		// 'i' will either be 0 or 1

        // left and right sides
        indices.push_back(i * 4 + 0);
        indices.push_back(i * 4 + 1);
        indices.push_back(i * 4 + 3);
        indices.push_back(i * 4 + 2);

        //top and bottom sides
        indices.push_back(i * 2 + 0);
        indices.push_back(i * 2 + 1);
        indices.push_back(i * 2 + 5);
        indices.push_back(i * 2 + 4);

        //front and back sides
        indices.push_back(i + 0);
        indices.push_back(i + 2);
        indices.push_back(i + 6);
        indices.push_back(i + 4);
    }
    /*
    vector< float > verticesCubeOBJ;
    vec3 colorCube = vec3(0.9,0.3,0.8);
    vector< vec2 > uvsCube;
    vector< vec3 > normalsCube; // Won't be used at the moment.
    vector< unsigned int > indicesCubeOBJ;
    bool res = loadOBJ("cubo.obj", verticesCubeOBJ, uvsCube, normalsCube, indicesCubeOBJ);
    */
    square = new Shape(vertices, indices);
    //square2 = new Shape(vertices2, indices);
    /*
    vector< vec3 > verticesOBJ;
    vector< vec3 > colors;
    vector< vec2 > uvs;
    vector< vec3 > normals; // Won't be used at the moment.
    vector< unsigned int > indicesOBJ;
    //vec3 color = vec3(0.5, 0.5, 0.8);
    bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    sphere = new Shape(verticesOBJ, indicesOBJ);
    */
    //vector<particle> particles(particles_num);
    particleInit();

    //glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_STATIC_DRAW);
    
    // In OpenGL, the Z-Axis points out of the screen.
    // Put the cube 5 units away from the camera.
	transformCube.SetPosition(vec3(0.0f, -0.0f, 0.0f));
    cout  << ' ' << transformCube.Position()[0] << ',' << transformCube.Position()[1] << ',' << transformCube.Position()[2] << endl;
    cameraPosition.SetPosition(vec3(0.0f, 0.0f, 8.0f));


/*
    transformParticle1.SetPosition(randomPosition());
    transformParticle1.Translate(vec3(0, 0, -5));
*/

	// Compile the vertex shader.
	vertexShader.InitFromFile("vertex.glsl", GL_VERTEX_SHADER);

	// Load and compile the fragment shader.
	fragmentShader.InitFromFile("fragment.glsl", GL_FRAGMENT_SHADER);

	// Create a shader program.
	shaderProgram = glCreateProgram();
	
	// Attach the vertex and fragment shaders to our program.
	vertexShader.AttachTo(shaderProgram);
	fragmentShader.AttachTo(shaderProgram);

	// Build shader program.
	glLinkProgram(shaderProgram);

	// After the program has been linked, we can ask it where it put our world matrix and camera matrix
	worldMatrixUniform = glGetUniformLocation(shaderProgram, "worldMatrix");
    cameraMatrixUniform = glGetUniformLocation(shaderProgram, "cameraView");


    //cout << "Use WASD to move, and the mouse to look around." << endl;
    cout << "Press escape to exit" << endl;

    //transform.SetPosition();
    cudaError_t err = cudaSuccess;
    cudaError_t cudaStatus;
	// Main Loop
	while (!glfwWindowShouldClose(window))
	{

        // Exit when escape is pressed
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            break;
        }

        // Calculate delta time.
        float dt = glfwGetTime();
        // Reset the timer.
        glfwSetTime(0);

        // Get the distance from the center of the screen that the mouse has moved
        glm::vec2 mouseMovement = mousePosition - (viewportDimensions / 2.0f);

        // Calculate the horizontal view angle
        float yaw = cameraPosition.Rotation().y;
        yaw += (int)mouseMovement.x * .001f;

        // Calculate the vertical view angle
        float pitch = cameraPosition.Rotation().x;
        pitch -= (int)mouseMovement.y * .001f;

        // Clamp the camera from looking up over 90 degrees.
        float halfpi = 3.1416 / 2.f;
        if (pitch < -halfpi) pitch = -halfpi;
        else if (pitch > halfpi) pitch = halfpi;

        // Set the new rotation of the camera.
        cameraPosition.SetRotation(glm::vec3(pitch, yaw, 0));
        

        // Move the cursor to the center of the screen
        glfwSetCursorPos(window, viewportDimensions.x/2, viewportDimensions.y/2);


        // Here we get some input, and use it to move the camera
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraPosition.Translate(cameraPosition.GetForward() * 5.0f * dt);
        }
        //if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            //transformCube.RotateY(-1.0f * dt);
            //transformParticle1.SetPosition(calculatePosition(transformParticle1.Position(), -0.5f * dt));
        //}
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraPosition.Translate(cameraPosition.GetForward() * -5.0f * dt);
        }

        /*if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            transformCube.RotateY(1.0f * dt);
        }*/
        
        // rotate square
        //transform.RotateY(1.0f * dt);



        // Cameras use a transformations matrix just like other renderable objects.
        // When we multiply by a world matrix, we are moving an object from local space to world space.
        // When using a camera, we do the exact opposite. We move everything else from world space into camera local space.
        // To do this we make a matrix that does the inverse of what a world matrix does.
        glm::mat4 viewMatrix = cameraPosition.GetInverseMatrix();

       
        // Perspective projection expands outward from the camera getting wider, and making things that are far away look smaller.
        /*
                    +-----------+
                    |           |
                    | O         |
        +----+      |         o |
        |*  .|      |           |
        +----+      +-----------+
        */
        // First we move everything
        float near = 1; // the nearest distance we will render anything
        float far = 10; // the furthest distance we will render anything.
        float width = 1; // width of the view in world space (usually maps directly to screen size)
        float height = viewportDimensions.y / viewportDimensions.x; // height of the view in world space (usually maps directly to screen size)
        
        // We do this by converting our coordinates into 
        glm::mat4 perspectiveProjection = glm::mat4(
            2/width, 0, 0, 0,                   // scale width down to fit in unit cube
            0, 2/height, 0, 0,                  // scale height
            0, 0, -(far+near)/(far-near), -1,   // scale depth, -1 converts our coordinates into homogeneous coordinates, which we need to keep our angles
            0, 0, (2*near*far)/(near-far), 1    // translate everything so that
            );

        // Compose view and projection into one matrix to send to the gpu
        glm::mat4 viewProjection = perspectiveProjection * viewMatrix;




        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.4, 0.4, 0.4, 0.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


		// Set the current shader program.
		glUseProgram(shaderProgram);

        // Send the camera matrix to the shader
        glUniformMatrix4fv(cameraMatrixUniform, 1, GL_FALSE, &(viewProjection[0][0]));
        

		// Draw using the worldMatrixUniform
        

        vec3 color2 = vec3(0.28f, 0.95f, 0.93f);
		//square2->Draw(shaderProgram,GL_QUADS, transformCube.GetMatrix(), worldMatrixUniform,color2);

        vec3 color3 = vec3(0.75f, 0.34f, 0.96f);
        //cout << &(particles[0].sphere) << endl;
        for (int i = 0; i < particles_num; i++) {
            (particlesShapes[i])->Draw(shaderProgram, GL_TRIANGLES, (particlesTransforms[i]).GetMatrix(), worldMatrixUniform);
        }
        vec3 color = vec3(0.9f, 0.27f, 0.89f);
        square->Draw(shaderProgram, GL_QUADS, transformCube.GetMatrix(), worldMatrixUniform);
        
        /*
        Aquí se llama el kernel
        */
        
        vec3* particlesPos = &particlesPosition[0];
        vec3* particlesVel = &particlesVelocity[0];

        vec3* particlesNewPos;
        vec3* particlesNewVel;

        particlesNewPos = (vec3*)malloc(sizeof(vec3) * particles_num);
        particlesNewVel = (vec3*)malloc(sizeof(vec3) * particles_num);

        vec3* d_Pos = NULL;
        err = cudaMalloc((void**)&d_Pos, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_NewPos = NULL;
        err = cudaMalloc((void**)&d_NewPos, sizeof(vec3) * particles_num);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_Vel = NULL;
        err = cudaMalloc((void**)&d_Vel, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_NewVel = NULL;
        err = cudaMalloc((void**)&d_NewVel, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(d_Pos, particlesPos, sizeof(vec3) * particles_num, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(d_Vel, particlesVel, sizeof(vec3)*particles_num, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        kernel_particula <<<1, particles_num>>>(particles_num, d_Pos, d_NewPos, d_Vel, d_NewVel, radio);
       
        cudaStatus = cudaDeviceSynchronize();

        
        err = cudaMemcpy(particlesNewPos, d_NewPos, sizeof(vec3)* particles_num, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(particlesNewVel, d_NewVel, sizeof(vec3) * particles_num, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        //vector<vec3> newVectorPor(begin(particlesNewPos), end(particlesNewPos));
        //vector<vec3> newVectorVel(begin(particlesNewVel), end(particlesNewVel));
        
        for (int i=0; i<particles_num;i++) {
            particlesPosition[i] = particlesNewPos[i];
            particlesVelocity[i] = particlesNewVel[i];
        }
        
        for (int i = 0; i < particles_num; i++) {
            
            //particlesTransforms[i].SetPosition(newPos);
            //particlesPosition[i] = particlesTransforms[i].Position();
            vec3 pos = particlesPosition[i];

            //particlesTransforms[i].Translate(vec3(0, 0, -5));
            if (abs(pos[0]) + radio > 1.0f) {
                particlesVelocity[i][0] = -particlesVelocity[i][0];
                pos[0] = (pos[0] / abs(pos[0]) * (1 - radio));
            }
            if (abs(pos[1]) + radio > 1.0f) {
                particlesVelocity[i][1] = -particlesVelocity[i][1];
                pos[1] = (pos[1] / abs(pos[1]) * (1 - radio)); 
            }
            if (abs(pos[2]) + radio > 1.0f) {
                particlesVelocity[i][2] = -particlesVelocity[i][2];
                pos[2] = (pos[2] / abs(pos[2]) * (1 - radio));

            }
            vec3 newPos = calculateNewPos(pos, particlesVelocity[i], dt, gravity);
            particlesPosition[i] = newPos;
            particlesTransforms[i].SetPosition(newPos);
            //cout << i << ' ' << particlesPosition[i][0] << ',' << particlesPosition[i][1] << ',' << particlesPosition[i][2] << endl;

        }
        err = cudaFree(d_Pos);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_Vel);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_NewPos);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_NewVel);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        free(particlesNewPos);
        free(particlesNewVel);
        

	    // Draw all indices in the index buffer
	    //glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_INT, (void*)0);

	    // Disable vertex attribute and unbind index buffer.
	    //glDisableVertexAttribArray(0);
	    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);



		// Stop using the shader program.
		glUseProgram(0);

		// Swap the backbuffer to the front.
		glfwSwapBuffers(window);

		// Poll input and window events.
		glfwPollEvents();

	}

	// Free memory from shader program and individual shaders
	glDeleteProgram(shaderProgram);


	// Free memory from shape object
	delete square;

	// Free GLFW memory.
	glfwTerminate();
    

	// End of Program.
	return 0;
}