Like the README says one has to go to https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-2-1 
and download the FBX Python SDK. For linux this is gz file. One has to uncompress it and follow the instructions that
are in the .txt file; when installation is finished one will have a `<fbx-installation-path>` where one installed. Then 
one continue with the instructions in https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html
from point 5 onward. Here is important to note that the documentation refers to the python bindings but one actually needs the
sdk files. Also: the `<yourFBXPythonSDKpath>\<version>` of the documentation is the same as `<fbx-installation-path>`.

Since I was using anaconda environments the path into one has to copy the files is: 
`<path-to-anaconda-installation>/envs/<env-name>/lib/python3.7/site-packages/`.