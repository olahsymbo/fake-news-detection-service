# Generated by Django 4.2.2 on 2023-06-17 14:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detector", "0002_alter_news_id"),
    ]

    operations = [
        migrations.AlterField(
            model_name="news",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="news",
            name="updated_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
