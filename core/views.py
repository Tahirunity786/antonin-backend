"""
    _summary_
    
    Module use to operate the Dashoboard Operations,
    include: Create project, Save project, Delete project, REST APIS
    Low Level: DB Operations
    
    Read Code 1 for more information.


"""


from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import os
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .utiles import extracter

# Create your views here.

@login_required(login_url="login")
def engine_helper(request):
    """_summary_
    Code 1

    Args:
        request (Http): Recieve a http/https respose and return a page

    Returns:
        HttpResponse: Returning a Dashboard page
    """
    return render(request, 'core/index.html')

class Engine(APIView):

    def post(self, request):
        uploaded_file = request.FILES.get('file')

        if uploaded_file:
            # Check if the file is a zip file
            if uploaded_file.name.endswith('.zip'):
                # Define the path where the file will be saved
                save_path = os.path.join(settings.MEDIA_ROOT, 'raw', uploaded_file.name)
                
                # Save the file
                path = default_storage.save(save_path, ContentFile(uploaded_file.read()))
                
                extracter(save_path, os.path.join(settings.MEDIA_ROOT, 'yeild'), uploaded_file.name)

                return Response({'message': 'File uploaded successfully'}, status=status.HTTP_201_CREATED)
            else:
                return Response({'error': 'Only zip files are allowed'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'No file was uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        