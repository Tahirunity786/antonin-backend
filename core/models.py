from django.db import models

# Create your models here.

class Projectfiles(models.Model):
    file = models.FileField(upload_to='result')

class Project(models.Model):
    title = models.CharField(max_length=100, db_index=True, default="")
    files = models.ManyToManyField(Projectfiles, db_index=True)
    date_created = models.DateTimeField(auto_now_add=True, db_index=True)

    
