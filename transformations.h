#pragma once

#include "glm/gtc/matrix_transform.hpp"

glm::mat4 identity();

glm::mat4 uniformScale(float s);

glm::mat4 translate(float tx, float ty, float tz);

glm::mat4 frustum(float left, float right, float bottom, float top, float near, float far);

glm::mat4 perspective(float fovy, float aspect, float near, float far);

glm::mat4 lookAt(glm::vec3 eye, glm::vec3 at, glm::vec3 up);