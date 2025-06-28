import mysql.connector as connector

class HotelBookingBackend:
    def __init__(self, host, user, database, password):
        self.host = host
        self.user = user
        self.database = database
        self.password = password
        self.mydb = connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def get_hotel_info(self, destination, date):
        cursor = self.mydb.cursor()
        query = "SELECT * FROM hotels_info destination = %s AND date = %s"
        cursor.execute(query, (destination, date))
        rows = cursor.fetchall()
        
        flight_info = "Below are the available Hotels with all the required information:\n"
        column_names = [desc[0] for desc in cursor.description]
        
        for row in rows:
            row_info = ""
            for i, col in enumerate(column_names):
                row_info += f"{col}: {row[i]}, "
            flight_info += row_info + "\n"
        
        cursor.close()
        return flight_info
    
    def book_hotels(self, passenger_id, destination, hotel_name, airline, date, number_of_room):
        cursor = self.mydb.cursor()
        get_hotel_query = "SELECT * FROM hotel_info WHERE  destination = %s AND hotel_name = %s AND date = %s"
        cursor.execute(get_hotel_query, ( destination, hotel_name, date ))
        hotel_info = cursor.fetchone()
        
        if not hotel_info or hotel_info[-1] < number_of_room:
            return "Sorry, all rooms are booked. Sorry for the inconvenience."
        
        try:
            self.mydb.start_transaction()
            update_hotel_query = "UPDATE hotel_info SET rooms = %s WHERE hotel_id = %s AND date = %s"
            cursor.execute(update_hotel_query, ( hotel_info[0] - number_of_room, hotel_info[0] ,date))
            
            insert_booking_query = "INSERT INTO bookings (passenger_id, hotel_id, rooms) VALUES (%s, %s, %s)"
            cursor.execute(insert_booking_query, (passenger_id, hotel_info[0], number_of_room))
            
            self.mydb.commit()
        except connector.Error as e:
            self.mydb.rollback()
            return f"Sorry, there was an error while booking the ticket: {str(e)}"
        finally:
            cursor.close()
        
        return "Ticket booked successfully"
    
    def cancel_booking(self, hotel_id, passenger_id):
        try:
            cursor = self.mydb.cursor()
            get_hotel_query = "SELECT * FROM hotel_info WHERE hotel_id = %s"
            get_booking_query = "SELECT * FROM bookings WHERE hotel_id = %s AND passenger_id = %s"
            
            cursor.execute(get_hotel_query, (hotel_id,))
            hotel_info = cursor.fetchone()
            
            cursor.execute(get_booking_query, (hotel_id, passenger_id))
            booking_info = cursor.fetchone()
            
            if not  hotel_info or not booking_info:
                return "Booking not found."

            self.mydb.start_transaction()
            update_hotel_query = "UPDATE hotel_info SET rooms =  %s WHERE hotel_id = %s"
            delete_booking_query = "DELETE FROM bookings WHERE hotel_id = %s AND passenger_id = %s"
            
            cursor.execute(update_hotel_query, (booking_info[-1], hotel_id))
            cursor.execute(delete_booking_query, (hotel_id, passenger_id))
            
            self.mydb.commit()
        except connector.Error as e:
            self.mydb.rollback()
            return f"There was an error while cancelling the ticket: {str(e)}"
        finally:
            cursor.close()
        
        return "Ticket canceled successfully"
