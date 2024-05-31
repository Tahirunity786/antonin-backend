from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from processor.views import login_render,logout_agent, register_render, account_render, savedproject_render, register_agent, login_agent

urlpatterns = [
    path('admin/', admin.site.urls),
    # Module Attachment
    path('', include('core.urls')),
    # Render redirection
    path('login/', login_render, name="login"),  
    path('register/', register_render, name="register"), 
    path('account/', account_render, name="account"), 
    path('saved_project/', savedproject_render, name="Savedproject"), 
    path('logout/', logout_agent, name="logout"), 
    # APIS
    path('login-me/', login_agent, name="loginme"), 
    path('register-me/', register_agent, name="registerme"), 

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
