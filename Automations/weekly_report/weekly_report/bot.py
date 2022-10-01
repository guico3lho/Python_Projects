"""
WARNING:

Please make sure you install the bot with `pip install -e .` in order to get all the dependencies
on your Python environment.

Also, if you are using PyCharm or another IDE, make sure that you use the SAME Python interpreter
as your IDE.

If you get an error like:
```
ModuleNotFoundError: No module named 'botcity'
```

This means that you are likely using a different Python interpreter than the one used to install the bot.
To fix this, you can either:
- Use the same interpreter as your IDE and install your bot with `pip install -e .`
- Use the same interpreter as the one used to install the bot (`pip install -e .`)

Please refer to the documentation for more information at https://documentation.botcity.dev/
"""

from botcity.core import DesktopBot


# Uncomment the line below for integrations with BotMaestro
# Using the Maestro SDK
# from botcity.maestro import *


class Bot(DesktopBot):
    def action(self, execution=None):
        # Fetch the Activity ID from the task:
        # task = self.maestro.get_task(execution.task_id)
        # activity_id = task.activity_id

        # Clear folder

        self.type_windows()
        self.paste('Downloads')
        
        if not self.find( "Downloads", matching=0.97, waiting_time=10000):
            self.not_found("Downloads")
        self.click()
        
        self.enter()

        if not self.find( "User-task", matching=0.97, waiting_time=10000):
            self.not_found("User-task")
        self.click()

        self.hold_shift()
        self.type_up()
        self.release_shift()
        self.delete()
        
        


        # Opens browser
        self.execute('C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk')

        if not self.find( "Select TMT", matching=0.97, waiting_time=10000):
            self.not_found("Select TMT")
        self.click()



        
        # **Project-Task-Description:**
        self.paste('https://app.clockify.me/shared/6337a74111ba64540842c426')

        self.enter()
        self.wait(5000)
        
        if not self.find( "Click Export", matching=0.97, waiting_time=10000):
            self.not_found("Click Export")
        self.click()

        if not self.find( "Save PDF", matching=0.97, waiting_time=10000):
            self.not_found("Save PDF")
        self.click()

        self.wait(5000)


        # **User-Task-Description:**
          
        if not self.find( "New url", matching=0.97, waiting_time=10000):
            self.not_found("New url")
        self.click()
        
        
        self.paste('https://app.clockify.me/shared/6337a88dbcb2b20745873bad')
        self.enter()

        self.wait(5000)

        if not self.find( "Click Export", matching=0.97, waiting_time=10000):
            self.not_found("Click Export")
        self.click()

        if not self.find( "Save PDF", matching=0.97, waiting_time=10000):
            self.not_found("Save PDF")
        self.click()

        self.wait(5000)
        self.type_windows()
        self.paste('Telegram')
        
        if not self.find( "Telegram", matching=0.97, waiting_time=10000):
            self.not_found("Telegram")
        self.click()
        
        
        if not self.find( "Find search", matching=0.97, waiting_time=10000):
            self.not_found("Find search")
        self.click()
        self.wait(3000)
        self.paste('Thales')

        
        if not self.find( "Find thales", matching=0.97, waiting_time=10000):
            self.not_found("Find thales")
        self.click()
       
        

        # self.paste('Saved')
        #
        # if not self.find( "Saved messages", matching=0.97, waiting_time=10000):
        #     self.not_found("Saved messages")
        # self.click()
        


        if not self.find( "Anexo", matching=0.97, waiting_time=10000):
            self.not_found("Anexo")
        self.click()
        
        self.wait(5000)
        
        # if not self.find( "Downloads", matching=0.97, waiting_time=10000):
        #     self.not_found("Downloads")
        # self.click()
        
        if not self.find( "User-task", matching=0.97, waiting_time=10000):
            self.not_found("User-task")
        self.click()

        self.hold_shift()
        self.type_up()
        self.release_shift()

        
        if not self.find( "Abrir", matching=0.97, waiting_time=10000):
            self.not_found("Abrir")
        self.click()
        
        
        
        if not self.find( "Comment", matching=0.97, waiting_time=10000):
            self.not_found("Comment")
        self.click()

        self.paste('Bom dia! Relatório semanal')
        
        if not self.find( "Send!", matching=0.97, waiting_time=10000):
            self.not_found("Send!")
        self.click()
        
        
        
        
        
        
        
        
        
     
      
      
        
        
        
        
        
        
        




        



            
   
        

        # Uncomment to mark this task as finished on BotMaestro
        # self.maestro.finish_task(
        #     task_id=execution.task_id,
        #     status=AutomationTaskFinishStatus.SUCCESS,
        #     message="Task Finished OK."
        # )


def not_found(self, label):
    print(f"Element not found: {label}")


if __name__ == '__main__':
    Bot.main()







