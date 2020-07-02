import java.awt.*;
public class Window extends Frame
{
	public Window(int width, int height)
	{
		//Create a panel
		Tetris panel = new Tetris(width,height);
		
		//Standard window sizes and conditions
		setTitle ("testTetris" ) ;
		setSize (1000, 900) ; 
		setLocation ( 100, 100) ; 
		setResizable( true ) ; 
		add( panel ) ;
		setVisible (true) ; 
	}
	
}
