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
import selenium

# Uncomment the line below for integrations with BotMaestro
# Using the Maestro SDK
# from botcity.maestro import *


class Bot(DesktopBot):

    def action(self, execution=None):
        # Fetch the Activity ID from the task:
        # task = self.maestro.get_task(execution.task_id)
        # activity_id = task.activity_id

        # Opens the csgoempire website.

        print("Automating csgoempire.com")
        self.browse("https://csgoempire.com/")

        if not self.find("Enter amount", matching=0.97, waiting_time=10000):
            self.not_found("Enter amount")
        self.click()

        if not self.find("Bet", matching=0.97, waiting_time=10000):
            self.not_found("Bet")
        self.click()

        antes_aposta = self.getValue()
        print("Antes da aposta: ",antes_aposta)

        if not self.find("Bet CT", matching=0.97, waiting_time=10000):
            self.not_found("Bet CT")
        self.click()
        conseguiu_apostar = self.getValue()

        while (conseguiu_apostar < antes_aposta):
            self.wait(5000)
            if not self.find("Bet CT", matching=0.97, waiting_time=10000):
                self.not_found("Bet CT")
            self.click()


        self.wait(25000)

        depois_aposta = self.getValue()
        print("Depois da aposta: ", depois_aposta)

        apostas = 5
        for i in range(apostas):
            print(f"aposta {i}")
            if(depois_aposta > antes_aposta):
                print("Ganhou")

                print("Antes da aposta: ", antes_aposta)
                if not self.find("Clear", matching=0.97, waiting_time=10000):
                    self.not_found("Clear")
                self.click()

                if not self.find("Bet", matching=0.97, waiting_time=10000):
                    self.not_found("Bet")
                self.click()

                if not self.find("Bet CT", matching=0.97, waiting_time=10000):
                    self.not_found("Bet CT")

                self.click()

                conseguiu_apostar = self.getValue()

                while (conseguiu_apostar < antes_aposta):
                    self.wait(5000)
                    if not self.find("Bet CT", matching=0.97, waiting_time=10000):
                        self.not_found("Bet CT")
                    self.click()

                self.wait(25000)

                depois_aposta = self.getValue()
                print("Depois da aposta: ", depois_aposta)

            else:
                print("Perdeu")
                print("Antes da aposta: ", antes_aposta)
                if not self.find("Bet X2", matching=0.97, waiting_time=10000):
                    self.not_found("Bet X2")
                self.click()

                if not self.find("Bet CT", matching=0.97, waiting_time=10000):
                    self.not_found("Bet CT")
                self.click()
                conseguiu_apostar = self.getValue()

                while (conseguiu_apostar < antes_aposta):
                    self.wait(5000)
                    if not self.find("Bet CT", matching=0.97, waiting_time=10000):
                        self.not_found("Bet CT")
                    self.click()

                self.wait(25000)

                depois_aposta = self.getValue()
                print("Depois da aposta: ", depois_aposta)


            antes_aposta = self.getValue()





        # if not self.find("Bet T", matching=0.97, waiting_time=10000):
        #     self.not_found("Bet T")
        # self.click()




        print("\nAutomation finished")


    def getValue(self):
            x = self.get_last_x()
            y = self.get_last_y()
            self.mouse_move(x=1756,y=159) # left
            self.mouse_down()
            self.mouse_move(x=1778,y=159) # right
            self.mouse_up()
            self.control_c()
            valor = float(self.get_clipboard().replace(',','.'))
            return valor




    def not_found(self, label):
        print(f"Element not found: {label}")


if __name__ == '__main__':
    Bot.main()



    # self.wait(5000)
    # x = self.get_last_x()
    # y = self.get_last_y()

    # print(f"Right side: {x},{y}",x,y)


# Uncomment to mark this task as finished on BotMaestro
        # self.maestro.finish_task(
        #     task_id=execution.task_id,
        #     status=AutomationTaskFinishStatus.SUCCESS,
        #     message="Task Finished OK."
        # )