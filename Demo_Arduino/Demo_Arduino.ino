//驱动OLED0.96所需的库
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
 
#define SCREEN_WIDTH 128 // 设置OLED宽度,单位:像素
#define SCREEN_HEIGHT 64 // 设置OLED高度,单位:像素

//自定义重置引脚  Adafruit_SSD1306库文件必需
#define OLED_RESET 4
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#include <SoftwareSerial.h>
// 设置蓝牙串口使用的针脚
SoftwareSerial BT(10, 11); // Pin10接HC05的TX Pin1接HC05的RX

#define red_led_pin 6 //RGB-LED引脚R
#define green_led_pin 5 //RGB-LED引脚G
#define blue_led_pin 3 //RGB-LED引脚B

int red_led = 0; //R Led 亮度
int green_led = 0; //G Led 亮度
int blue_led = 0; //B Led 亮度

int color_change_delay = 12;     //颜色改变速度
unsigned long start_time = millis();    //获取开始时间

char var;
int W_time = 0;
int N1_time = 0;
int N2_time = 0;
int N3_time = 0;
int REM_time = 0;

void setup()
{
    //RGB LED初始化部分
    Serial.begin(9600);
    
    //设置引脚为相应工作模式
    pinMode(red_led_pin, OUTPUT);
    pinMode(green_led_pin, OUTPUT);
    pinMode(blue_led_pin, OUTPUT);

    Serial.println("Please Input RGB value.");
    Serial.println(start_time);
    
    //初始值
    red_led = 10;
    green_led = 10;
    blue_led = 10;
    light_led(red_led, green_led, blue_led);

    //OLED初始化部分
    // 初始化OLED并设置其IIC地址为 0x3C
    display.begin(SSD1306_SWITCHCAPVCC, 0x3C);

    //蓝牙初始化部分
    BT.begin(9600); //设定软串口波特率

    time_display(W_time, N1_time, N2_time, N3_time, REM_time);
    display.display();
}

void loop(){

//    unsigned long current_time = millis();
//    Serial.println("current_time: ");
//    Serial.println(current_time);
    delay(1000);
    //while(Serial.available() > 0)      //当有串口发来信号（硬接口）
    while(BT.available())   //当有串口发来信号（软接口 蓝牙）
    {
        var = BT.read();
        Serial.print(var);
        if(var == '0')                //传过来的label是0，清醒期，显示橙色
        {
            fade_change_light(255, 0, 0);
            W_time += 1;
        }
        if(var == '1')                //传过来的label是1，N1阶段，显示蓝色1
        {
            fade_change_light(100, 149, 237);
            N1_time += 1;
        }
        if(var == '2')                //传过来的label是2，N2阶段，显示蓝色2
        {
            fade_change_light(65, 105, 225);
            N2_time += 1;
        }
        if(var == '3')                //传过来的label是3，N3阶段，显示蓝色3
        {
            fade_change_light(0, 0, 128);
            N3_time += 1;
        }
        if(var == '4')                //传过来的label是4，REM阶段，显示紫色
        {
            fade_change_light(148, 0, 211);
            REM_time += 1;
        }


    }        
    //OLED显示
    time_display(W_time, N1_time, N2_time, N3_time, REM_time);
    display.display();
}


void fade_change_light(int new_red_value, int new_green_value,int new_blue_value)
//颜色渐变
{
    int fade_step = 1;  //fade_step和color_change_delay只要定义一个，所以fade_step取1
    while((red_led != new_red_value)||(green_led != new_green_value)||(blue_led != new_blue_value))
    {
        if(red_led < new_red_value)
        {
            red_led += fade_step;
        }
        else if(red_led > new_red_value)
        {
            red_led -= fade_step;
        }
        if(green_led < new_green_value)
        {
            green_led += fade_step;
        }
        else if(green_led > new_green_value)
        {
            green_led -= fade_step;
        }
        if(blue_led < new_blue_value)
        {
            blue_led += fade_step;
        }
        else if(blue_led > new_blue_value)
        {
            blue_led -= fade_step;
        }
        analogWrite(red_led_pin, red_led);
        delay(color_change_delay);
        analogWrite(green_led_pin, green_led);
        delay(color_change_delay);
        analogWrite(blue_led_pin, blue_led);
        delay(color_change_delay);
    }
}


void light_led(int red_value, int green_value, int blue_value)
{
    //最好同时更改这三个全局变量
    red_led = red_value;
    green_led = green_value;
    blue_led = blue_value;
    analogWrite(red_led_pin, red_led);
    delay(5);
    analogWrite(green_led_pin, green_led);
    delay(5);
    analogWrite(blue_led_pin, blue_led); 
    delay(5); 

}

void time_display(int W, int N1, int N2, int N3, int REM)
{
    // 清除屏幕
    display.clearDisplay();

    // 设置字体颜色
    display.setTextColor(WHITE);

    //设置字体大小
    display.setTextSize(1.5);

    //设置光标位置
    display.setCursor(0, 0);
    display.print("each stage lasts:");

    //display.setTextSize(1.5);
    display.setCursor(0, 10);
    display.print("Wake: ");
    display.print(W*0.5);
    display.print(" min");

    display.setCursor(0, 20);
    display.print("NREM1: ");
    display.print(N1*0.5);
    display.print(" min");

    display.setCursor(0, 30);
    display.print("NREM2: ");
    display.print(N2*0.5);
    display.print(" min");

    display.setCursor(0, 40);
    display.print("NREM3: ");
    display.print(N3*0.5);
    display.print(" min");

    display.setCursor(0, 50);
    display.print("REM: ");
    display.print(REM*0.5);
    display.print(" min");
}
