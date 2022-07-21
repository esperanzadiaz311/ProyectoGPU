/*
Title: Drawing a Cube
File Name: fragment.glsl
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


#version 400 core

//in vec3 newColor;
in vec3 fragColor;
in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform vec3 La;
uniform vec3 Ld;
uniform vec3 Ls;
uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;
uniform uint shininess;
uniform float constantAttenuation;
uniform float linearAttenuation;
uniform float quadraticAttenuation;

out vec4 outColor;
void main()
{
    vec3 ambient = Ka * La;

    // diffuse
    // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
    vec3 normalizedNormal = normalize(fragNormal);
    vec3 toLight = lightPosition - fragPosition;
    vec3 lightDir = normalize(toLight);
    float diff = max(dot(normalizedNormal, lightDir), 0.0);
    vec3 diffuse = Kd * Ld * diff;

    // specular
    vec3 viewDir = normalize(fragPosition);
    vec3 reflectDir = reflect(-lightDir, normalizedNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = Ks * Ls * spec;

    // attenuation
    float distToLight = length(toLight);
    float attenuation = constantAttenuation
        + linearAttenuation * distToLight
        + quadraticAttenuation * distToLight * distToLight;

    vec3 result = (ambient + ((diffuse + specular) / attenuation)) * fragColor;
    outColor = vec4(result, 1.0);
}