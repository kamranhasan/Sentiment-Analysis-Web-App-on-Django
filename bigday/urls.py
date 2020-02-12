from django.conf.urls import url, include
from django.contrib import admin
from wedding import views
from django.conf import settings
from django.conf.urls.static import static
# from facial_emotion_image import emotional
admin.autodiscover()
urlpatterns = [
    url(r'^admin/',include(admin.site.urls)),
    url(r'^', views.simple_upload, name='simple_upload'),
    url(r'^', include('wedding.urls')),
    # url(r'^',views.emotional,name='emotional'),
    # url(r'^', include('guests.urls')),
    url('^accounts/', include('django.contrib.auth.urls'))
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
    