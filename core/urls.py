from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import Engine, engine_helper

urlpatterns = [
    path('', engine_helper, name="dashboard"),
    path('engine', Engine.as_view(), name="workerapi")
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)