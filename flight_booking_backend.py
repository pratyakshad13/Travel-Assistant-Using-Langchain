import mysql.connector as connector

class FlightBookingBackend:
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
    
    def get_flight_info(self, departure, destination, date):
        cursor = self.mydb.cursor()
        query = "SELECT * FROM flights_info WHERE departure = %s AND destination = %s AND date = %s"
        cursor.execute(query, (departure, destination, date))
        rows = cursor.fetchall()
        
        flight_info = "Below are the available flights with all the required information:\n"
        column_names = [desc[0] for desc in cursor.description]
        
        for row in rows:
            row_info = ""
            for i, col in enumerate(column_names):
                row_info += f"{col}: {row[i]}, "
            flight_info += row_info + "\n"
        
        cursor.close()
        return flight_info
    
    def book_flight(self, departure, passenger_id, destination, airport, airline, date, seats):
        cursor = self.mydb.cursor()
        get_flight_query = "SELECT * FROM flights_info WHERE departure = %s AND destination = %s AND airport = %s AND date = %s AND airline = %s"
        cursor.execute(get_flight_query, (departure, destination, airport, date, airline))
        flight_info = cursor.fetchone()
        
        if not flight_info or flight_info[-1] < seats:
            return "Sorry, all tickets are booked. Sorry for the inconvenience."
        
        try:
            self.mydb.start_transaction()
            update_flight_query = "UPDATE flights_info SET seats = seats - %s WHERE flight_id = %s AND date = %s"
            cursor.execute(update_flight_query, (seats, flight_info[0], date))
            
            insert_booking_query = "INSERT INTO bookings (passenger_id, flight_id, seats) VALUES (%s, %s, %s)"
            cursor.execute(insert_booking_query, (passenger_id, flight_info[0], seats))
            
            self.mydb.commit()
        except connector.Error as e:
            self.mydb.rollback()
            return f"Sorry, there was an error while booking the ticket: {str(e)}"
        finally:
            cursor.close()
        
        return "Ticket booked successfully"
    
    def cancel_booking(self, flight_id, passenger_id):
        try:
            cursor = self.mydb.cursor()
            get_flight_query = "SELECT * FROM flight_info WHERE flight_id = %s"
            get_booking_query = "SELECT * FROM bookings WHERE flight_id = %s AND passenger_id = %s"
            
            cursor.execute(get_flight_query, (flight_id,))
            flight_info = cursor.fetchone()
            
            cursor.execute(get_booking_query, (flight_id, passenger_id))
            booking_info = cursor.fetchone()
            
            if not flight_info or not booking_info:
                return "Booking not found."

            self.mydb.start_transaction()
            update_flight_query = "UPDATE flight_info SET seats = seats + %s WHERE flight_id = %s"
            delete_booking_query = "DELETE FROM bookings WHERE flight_id = %s AND passenger_id = %s"
            
            cursor.execute(update_flight_query, (booking_info[-1], flight_id))
            cursor.execute(delete_booking_query, (flight_id, passenger_id))
            
            self.mydb.commit()
        except connector.Error as e:
            self.mydb.rollback()
            return f"There was an error while cancelling the ticket: {str(e)}"
        finally:
            cursor.close()
        
        return "Ticket canceled successfully"
