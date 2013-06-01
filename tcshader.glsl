#version 400

layout ( vertices = 3 ) out;  

uniform int inner_level;
uniform int outer_level;      

void main ()
{                
    gl_out [gl_InvocationID].gl_Position = gl_in [gl_InvocationID].gl_Position;
    
    if ( gl_InvocationID == 0 )        
    {
        gl_TessLevelInner [0] = inner_level;
        gl_TessLevelOuter [0] = outer_level;
        gl_TessLevelOuter [1] = outer_level;
        gl_TessLevelOuter [2] = outer_level;
    }
}