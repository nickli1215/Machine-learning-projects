import java.awt.*;
import java.util.Arrays;
public class Storage 
{
	//each board is used to paint the next 3 pieces on the board
	public int[][] board1 = new int[5][5], board2 = new int[5][5], board3 = new int[5][5];
	// holds the next 3 pieces
	private Piece piece1, piece2, piece3;
	// piece will be in the centre of the 5x5 board for all pieces
	final Point centre = new Point(2,2);
	//color array
	Color[] colors = new Color[] {null, Color.orange, Color.blue, Color.red, Color.green,
	Color.MAGENTA, Color.YELLOW, Color.cyan};
	//constructor
	public Storage ()
	{
		//creates 3 new random pieces, and copies pieces to their boards
		piece1 = new Piece(centre,(int)(Math.random()*7),0);
		piece2 = new Piece(centre,(int)(Math.random()*7),0);
		piece3 = new Piece(centre,(int)(Math.random()*7),0);
		piece1.copy(board1);
		piece2.copy(board2);
		piece3.copy(board3);
	}
	//copies the values of array1 to array2
	public void Arraycopy(int[][] i, int[][]j)
	{
		for (int x =0; x<i.length; x++)
			for (int y = 0; y<i[x].length; y++)
				i[x][y] = j[x][y];
	}
	public int shift()
	{
		int shape = piece1.getShape();
		//clear and refill 1st next shape
		piece1 = piece2;
		Arraycopy(board1, board2);
		//clear and refill 2nd next shape
		piece2 = piece3;
		Arraycopy(board2, board3);
		// clear and create a new random 3rd shape;
		for (int x = 0; x<board3.length; x++)
			for (int y = 0; y<board3.length; y++)
				board3[x][y] = 0;
		piece3 = new Piece(centre,(int)(Math.random()*7),0);
		piece3.copy(board3);
		return shape;
	}

	public void display(Graphics g, int size, int width)
	{
		
		for (int i = 0; i<5; i++)
		{
			for (int j = 0; j<5; j++)
			{
				//fill in piece1
				if(board1[i][j]>0)
				{
					g.setColor(colors[board1[i][j]]);
					g.fillRect((j+width)*size, (i+6)*size, size, size);
					g.setColor(Color.black);
				}
				//fill in piece2
				if(board2[i][j]>0)
				{
					g.setColor(colors[board2[i][j]]);
					g.fillRect((j+width)*size, (i+12)*size, size, size);
					g.setColor(Color.black);
				}
				//fill in piece3
				if(board3[i][j]>0)
				{
					g.setColor(colors[board3[i][j]]);
					g.fillRect((j+width)*size, (i+18)*size, size, size);
					g.setColor(Color.black);
				}
				//draw board
				g.drawRect((j+width)*size, (i+6)*size, size, size);
				g.drawRect((j+width)*size, (i+12)*size, size, size);
				g.drawRect((j+width)*size, (i+18)*size, size, size);
			}
		}
		
	}
}
