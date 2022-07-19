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
//#include <shader.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <vector>
#include "shape.h"
#include "transform3d.h"
#include "shader2.h"

using namespace glm;
using namespace std;

/*
    Load an OBJ file. Only triangular faces.
*/

bool loadOBJ(
    const char * path,
    vector < vec3 > & out_vertices,
    vector < vec2 > & out_uvs,
    vector < vec3 > & out_normals,
    vector < unsigned int > & out_indexs
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
        out_vertices.push_back(vertex);
        
    }
}

GLFWwindow* window;

Shape* square;
Shape* square2;
Shape* sphere;

// The transform being used to draw our shape
Transform3D transform;

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
	vector<vec3> vertices;
    vertices.push_back(vec3(-1, -1, -1));
	vertices.push_back(vec3(-1, -1, 1));
	vertices.push_back(vec3(-1, 1, -1));
	vertices.push_back(vec3(-1, 1, 1));
    vertices.push_back(vec3(1, -1, -1));
    vertices.push_back(vec3(1, -1, 1));
    vertices.push_back(vec3(1, 1, -1));
    vertices.push_back(vec3(1, 1, 1));

    vector<vec3> vertices2;
    vertices2.push_back(vec3(-0.5, -0.5, -0.5));
	vertices2.push_back(vec3(-0.5, -0.5, 0.5));
	vertices2.push_back(vec3(-0.5, 0.5, -0.5));
	vertices2.push_back(vec3(-0.5, 0.5, 0.5));
    vertices2.push_back(vec3(0.5, -0.5, -0.5));
    vertices2.push_back(vec3(0.5, -0.5, 0.5));
    vertices2.push_back(vec3(0.5, 0.5, -0.5));
    vertices2.push_back(vec3(0.5, 0.5, 0.5));

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
    square = new Shape(vertices, indices);
    square2 = new Shape(vertices2, indices);
    
    vector< vec3 > verticesOBJ;
    vector< vec3 > colors;
    vector< vec2 > uvs;
    vector< vec3 > normals; // Won't be used at the moment.
    vector< unsigned int > indicesOBJ;
    //vec3 color = vec3(0.5, 0.5, 0.8);
    bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    sphere = new Shape(verticesOBJ, indicesOBJ);
    //glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_STATIC_DRAW);
    
    // In OpenGL, the Z-Axis points out of the screen.
    // Put the cube 5 units away from the camera.
	transform.SetPosition(vec3(0, 0, -5));

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


    cout << "Use WASD to move, and the mouse to look around." << endl;
    cout << "Press escape to exit" << endl;




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
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            transform.RotateY(-1.0f * dt);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraPosition.Translate(cameraPosition.GetForward() * -5.0f * dt);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            transform.RotateY(1.0f * dt);
        }


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
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


		// Set the current shader program.
		glUseProgram(shaderProgram);

        // Send the camera matrix to the shader
        glUniformMatrix4fv(cameraMatrixUniform, 1, GL_FALSE, &(viewProjection[0][0]));
		
		// Draw using the worldMatrixUniform
        vec3 color = vec3(0.9f, 0.27f, 0.89f);
		square->Draw(shaderProgram,GL_QUADS, transform.GetMatrix(), worldMatrixUniform,color);

        vec3 color2 = vec3(0.28f, 0.95f, 0.93f);
		//square2->Draw(shaderProgram,GL_QUADS, transform.GetMatrix(), worldMatrixUniform,color2);

        vec3 color3 = vec3(0.75f, 0.34f, 0.96f);
        sphere->Draw(shaderProgram, GL_TRIANGLES, transform.GetMatrix(), worldMatrixUniform, color3);

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