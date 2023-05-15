# Generated by Django 3.0.5 on 2023-05-07 17:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hospital', '0018_auto_20201015_2036'),
    ]

    operations = [
        migrations.CreateModel(
            name='Diagnosis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patientId', models.PositiveIntegerField(null=True)),
                ('test', models.TextField(max_length=500)),
                ('doctorsAdvice', models.TextField(max_length=500)),
                ('prescription', models.TextField(max_length=500)),
            ],
        ),
    ]
