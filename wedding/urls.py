from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$',views.img_submit,name='img_submit'),
    url(r'^$', views.home, name='home'),
    
]
