/*
Title: Drawing a Cube
File Name: vertex.glsl
Copyright ? 2016
Author: David Erbelding
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#version 330 core


// vertex position attribute
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

//out vec3 fragColor;
out vec3 fragPosition;
out vec3 fragNormal;
// uniform variables

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 transform;

void main()
{
    fragPosition = vec3(model * vec4(position, 1.0));
    //fragColor = color;
    fragNormal = mat3(transpose(inverse(model))) * normal;

    gl_Position = projection * view * transform * vec4(fragPosition, 1.0);
}