from django import forms
from .models import Call_History, ProblemStatus

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = Call_History
        # fields = ['employeeID', 'customerID', 'record', 'problem_summary', 'problem_status']
        fields = ['employeeID', 'customerID', 'record']
        widgets = {
            'employeeID': forms.Select(),
            'customerID': forms.Select(),
            # 'problem_status': forms.Select(choices=ProblemStatus.choices),
        }
    
    def __init__(self, *args, **kwargs):
        super(FileUploadForm, self).__init__(*args, **kwargs)
        self.fields['employeeID'].empty_label = "Employee"
        self.fields['customerID'].empty_label = "Customer"
        # self.fields['problem_status'].empty_label = "Problem Status"