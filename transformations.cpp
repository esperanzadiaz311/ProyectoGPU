#pragma once

#include "transformations.h"
#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <numbers>

glm::mat4 identity() {
	glm::mat4 identidad = glm::mat4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);
	return identidad;
}

glm::mat4 uniformScale(float s) {
	glm::mat4 matrix = glm::mat4(
		s, 0, 0, 0,
		0, s, 0, 0,
		0, 0, s, 0,
		0, 0, 0, 1
	);
	return matrix;
}

glm::mat4 translate(float tx, float ty, float tz) {
	glm::mat4 matrix = glm::mat4(
		1, 0, 0, tx,
		0, 1, 0, ty,
		0, 0, 1, tz,
		0, 0, 0, 1
	);
	return matrix;
}

glm::mat4 frustum(float left, float right, float bottom, float top, float near, float far) {
	float rl = right - left;
	float tb = top - bottom;
	float fn = far - near;
	glm::mat4 matrix = glm::mat4(
		(2 * near / rl), 0, (right + left) / rl, 0,
		0, (2 * near / tb), (top + bottom) / tb, 0,
		0, 0, -(far + near) / fn, -(2 * near * far / fn),
		0, 0, -1, 0
	);
	return matrix;
}

glm::mat4 perspective(float fovy, float aspect, float near, float far) {
	float halfHeight = glm::tan(M_PI * fovy / 360.0) * near;
	float halfWidth = halfHeight * aspect;
	glm::mat4 matrix = frustum(-halfWidth, halfWidth, -halfHeight, halfHeight, near, far);
	return matrix;
}

glm::mat4 lookAt(glm::vec3 eye, glm::vec3 at, glm::vec3 up) {
	glm::vec3 forward = at - eye;
	glm::vec3 forwardNormalized = glm::normalize(forward);

	glm::vec3 side = glm::cross(forwardNormalized, up);
	glm::vec3 normalizedSide = glm::normalize(side);

	glm::vec3 newUp = glm::cross(normalizedSide, forwardNormalized);
	glm::vec3 normalizedUp = glm::normalize(newUp);

	glm::mat4 matrix = glm::mat4(
		side[0], side[1], side[2], -(glm::dot(side, eye)),
		newUp[0], newUp[1], newUp[2], -(glm::dot(newUp, eye)),
		-forward[0], -forward[1], -forward[2], glm::dot(forward, eye),
		0, 0, 0, 1
	);
	return matrix;
}