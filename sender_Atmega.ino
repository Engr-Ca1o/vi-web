#include <SPI.h>
#include <LoRa.h>

String receivedData = ""; // Variable to store received UART data

void setup() {
  pinMode(7,OUTPUT);

  while (!Serial);

  if (!LoRa.begin(433E6)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }
}

void loop() {
  // Check if data is available on UART
  digitalWrite(7,HIGH);
  
  if (Serial.available() > 0) {
    receivedData = Serial.readStringUntil('\n'); // Read data until newline character

    // Send the received data over LoRa
    LoRa.beginPacket();
    LoRa.print(receivedData);
    LoRa.endPacket();
  }

  delay(500);
  digitalWrite(7,LOW);
  delay(500);// Small delay to avoid overwhelming the loop
}
