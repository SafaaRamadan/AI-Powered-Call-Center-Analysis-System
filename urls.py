from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('home/', views.home, name='home'),
    path('calls/', views.calls, name='calls'),
    path('call_detail/<int:call_id>/', views.call_detail, name='call_detail'),
    path('test/', views.call_test, name='call_test'),
    path('calls/employee/<int:employee_id>/', views.employee, name='employee'),
    path('upload_call/', views.upload_call, name='upload_call'),
    path('agent/<int:employee_id>/',views.agent_details, name='agent_details'),
]