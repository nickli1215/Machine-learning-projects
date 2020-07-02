import java.awt.event.*;
import java.util.Scanner;
public class Main 
{
	public static void main(String[] args)
	{
		//Taking in user input for dimensions of the board
		System.out.println("Enter your dimensions.");
		Scanner reader = new Scanner(System.in);
		int x = reader.nextInt(); int y = reader.nextInt();
		//Creating the program window, and using arguments x and y to represent dimensions
		Window frame = new Window(x,y);
		frame.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e)
			{
				System.exit(0);
			}
		});
	}
}
